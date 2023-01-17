import sys
sys.path.append("../usr/lib/")
'''
INPUT_ROOT = '/home/cszsolnai/Projects/kaggle_covid19_nlp/input'
OUTPUT_ROOT = '/home/cszsolnai/Projects/kaggle_covid19_nlp/output'
BIOBERT_ROOT = '/home/cszsolnai/Projects/kaggle_covid19_nlp/input/biobert_v1.1_pubmed'
'''

INPUT_ROOT = '/kaggle/input/CORD-19-research-challenge'
OUTPUT_ROOT = '.'
BIOBERT_ROOT = '/kaggle/input/biobert-pretrained/biobert_v1.1_pubmed'


METADATA = INPUT_ROOT + '/metadata.csv'
MODEL_PATH = BIOBERT_ROOT + '/model.ckpt-1000000'
VOCAB_FILE = BIOBERT_ROOT + '/vocab.txt'
BERT_CONFIG = BIOBERT_ROOT + '/bert_config.json'
!pip install tensorflow-gpu==1.14.0
!pip install bert-tensorflow
import pickle
import argparse
import re
import os
import glob
import json
import pandas as pd
from tqdm.notebook import tqdm
import nltk
import argparse
import re
import numpy as np
import pickle
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from bert_embeddings import bert_embedding_generator
nltk.download('stopwords')
def fold(e):
    if isinstance(e, list):
        return ' '.join(e)
    else:
        return e

def get_content(file_path):
    with open(file_path) as file:
        content = json.load(file)

        if 'abstract' in content:
            abstract = []
            for entry in content['abstract']:
                abstract.append(entry['text'])

            full_abstract = '\n'.join(abstract)
        else:
            full_abstract = ''

        # Body text
        articles_text = []
        if 'body_text' in content:
            for entry in content['body_text']:
                articles_text.append(entry['text'])
            full_text = '\n'.join(articles_text)
            concat_text = '\n'.join([full_abstract, full_text]).strip()
        else:
            full_text = ''
            concat_text = ''
            


        # Authors
        authors = []
        if 'authors' in content['metadata']:
            for author in content['metadata']['authors']:
                authors.append(' '.join([fold(author['first']), fold(author['middle']), fold(author['last'])]))

        return content['paper_id'], concat_text, content['metadata'][
            'title'], full_abstract, '\n'.join(authors), full_text


def lower_case(input_str):
    input_str = input_str.lower()
    return input_str
all_json = glob.glob(INPUT_ROOT + '/**/*.json', recursive=True)
# https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=887
query_text = "What is the best method to combat the hypercoagulable state seen in COVID-19 ?"
converted_query = next(bert_embedding_generator([query_text], [0], BERT_CONFIG, VOCAB_FILE, MODEL_PATH))
query_embeddings = np.stack([e[0] for e in converted_query.values()])
# Get the article with the closest embedding tot the query string embedding


class EmptyContent(Exception):
    pass


def flatten_list(values):
    l = []
    for e in values:
        if isinstance(e, list):
               for embedding in e:
                    l.append(embedding)
        else:
            l.append(e)
    return l


def average_query_distance(article_embeddings):
    avg_best_distance_from_query = 0
    for i in range(query_embeddings.shape[0]):
        avg_best_distance_from_query += np.min(np.linalg.norm(query_embeddings[i] - article_embeddings, axis=1))
    return avg_best_distance_from_query / query_embeddings.shape[0]


def article_content_generator(all_json):
    for idx, entry in tqdm(enumerate(all_json)):
        paper_id, concat_text, title, abstract, authors, _ = get_content(entry)
        if concat_text == '':
            yield EmptyContent()
        else:
            yield concat_text


closest_ids = []
closest_num = 10
closest_distances = []

update_frequency = 1000

for idx, converted_article in enumerate(bert_embedding_generator(article_content_generator(all_json), list(range(len(all_json))), BERT_CONFIG, VOCAB_FILE, MODEL_PATH)):
    if isinstance(converted_article, EmptyContent):
        continue
    
    article_embeddings = np.stack(flatten_list(converted_article.values()))
            
    closest_ids.append(idx)
    closest_distances.append(average_query_distance(article_embeddings))
    
    # Remove article id with the highest distance
    if len(closest_ids) > closest_num:
        max_id = np.argmax(closest_distances)
        del closest_distances[max_id]
        del closest_ids[max_id]

    if (idx+1) % update_frequency == 0:
        with open('checkpoint.pkl', 'wb') as f:
            data = {
                'idx': idx,
                'closest_ids': closest_ids,
                'closest_distances': closest_distances
            }
            pickle.dump(data, f)



with open('checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)
    idx = data['idx']
    closest_ids = data['closest_ids']
    closest_distances = data['closest_distances']
idxs = np.argsort(closest_distances)
closest_ids = np.array(closest_ids)[idxs]
closest_distances = np.array(closest_distances)[idxs]

selected_jsons = list(np.array(all_json)[idxs])

#selected_jsons = [all_json[0]]
import torch
from torch.utils.data import DataLoader, SequentialSampler
from collections import namedtuple
from transformers import BertTokenizer
from modeling_bertabs import BertAbs, build_predictor

from utils_bertabs import (
    CovidDataset,
    build_mask,
    compute_token_type_ids,
    encode_for_summarization,
    truncate_or_pad,
)
def build_data_iterator(all_jsons, args, tokenizer):
    dataset = CovidDataset(all_jsons)
    sampler = SequentialSampler(dataset)

    def collate_fn(data):
        return collate(data, tokenizer, block_size=512, device=args['device'])

    iterator = DataLoader(dataset, sampler=sampler, batch_size=args['batch_size'], collate_fn=collate_fn, )

    return iterator, dataset


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]
    summaries = [" ".join(summary_list) for _, _, summary_list in data]

    encoded_text = [encode_for_summarization(story, summary, tokenizer) for _, story, summary in data]
    encoded_stories = torch.tensor(
        [truncate_or_pad(story, block_size, tokenizer.pad_token_id) for story, _ in encoded_text]
    )
    encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(encoded_stories),
        src=encoded_stories.to(device),
        segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        tgt_str=summaries,
    )

    return batch

Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])

BATCH_SIZE = 1
MIN_LENGTH = 50
MAX_LENGTH = 200
BEAM_SIZE = 5
DEVICE = torch.device("cuda")

args = {
    'documents_dir': INPUT_ROOT,
    'summaries_output_dir': 'output_dir',
    'compute_rouge': False,
    'no_cuda': False,
    'batch_size': BATCH_SIZE,
    'min_length': MIN_LENGTH,
    'max_length': MAX_LENGTH,
    'beam_size': BEAM_SIZE,
    'alpha': 0.95,
    'block_trigram': True,
    'device': DEVICE
}
df = pd.read_csv(
    METADATA,
    usecols=["title", "abstract", "authors", "doi", "publish_time"],
)

# drop duplicates
# df=df.drop_duplicates()
df = df.drop_duplicates(subset="abstract", keep="first")
# drop NANs
df = df.dropna()

df.head()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
model.to(args['device'])
model.eval()

symbols = {
    "BOS": tokenizer.vocab["[unused0]"],
    "EOS": tokenizer.vocab["[unused1]"],
    "PAD": tokenizer.vocab["[PAD]"],
}

args['result_path'] = ""
args['temp_dir'] = ""

data_iterator, dataset = build_data_iterator(selected_jsons, args, tokenizer)

predictor = build_predictor(args, tokenizer, symbols, model)
all_summaries =  []

data = {'Paper id': [], 'Title': [], 'Summary': []}

for i, batch in enumerate(tqdm(data_iterator)):
    with open(selected_jsons[i]) as json_file:
        article = json.load(json_file)
    
    batch_data = predictor.translate_batch(batch)
    translations = predictor.from_batch(batch_data)
    data['Paper id'].append(article['paper_id'])
    data['Title'].append(article['metadata']['title'])
    data['Summary'].append(translations[0][2])
df = pd.DataFrame.from_dict(data)
df.to_csv(os.path.join(OUTPUT_ROOT, 'summaries.csv'))
