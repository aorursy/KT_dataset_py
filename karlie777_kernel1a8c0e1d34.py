## This cell installs a package, only needs to run once for all

## Uncomment the following line



!pip install sentence_transformers #commit的时候还是要跑这个 uncomment



#Added: to load NLI model

!pip install git+https://github.com/gsarti/covid-papers-browser/blob/master/requirements.txt

!python -m spacy download en_core_web_sm #Y

!python -m nltk.downloader punkt #Y

#!python scripts/download_model.py# --model scibert-nli   ###this cannot work
import urllib.request

import zipfile

import os

import math

import pickle

import prettytable



import pandas as pd

import numpy as np

import scipy

import textwrap



import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from sentence_transformers import models, losses, SentencesDataset, SentenceTransformer

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sentence_transformers.readers import *



import warnings

warnings.simplefilter('ignore')
#download a model fine-tuned on NLI from HuggingFace's cloud repository.

#Don't know how to get package/function from github yet

import os

import argparse

from shutil import rmtree

from transformers import AutoTokenizer, AutoModel

from sentence_transformers import models, SentenceTransformer



MODELS_PATH = "models"



MODELS_PRETRAINED = {

    'scibert': 'allenai/scibert_scivocab_cased',

    'biobert': 'monologg/biobert_v1.1_pubmed',

    'covidbert': ' deepset/covid_bert_base',

}



MODELS_FINETUNED = {

    'scibert-nli': 'gsarti/scibert-nli',

    'biobert-nli': 'gsarti/biobert-nli',

    'covidbert-nli': 'gsarti/covidbert-nli'

}



MODELS = {**MODELS_PRETRAINED, **MODELS_FINETUNED}





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model", 

        default="scibert-nli",  ####################################change here to load differnt nli model

        type=str, 

        required=False,

        help="Model selected in the list: " + 

             ", ".join(list(MODELS_PRETRAINED) + list(MODELS_FINETUNED))

    )

    parser.add_argument(

        "--do_lower_case", 

        action="store_true",

        help="Use a cased language model."

    )

    parser.add_argument(

        "--max_seq_length", 

        default=128, 

        type=int,

        required=False,

        help="Sequence length used by the language model."

    )

    #args = parser.parse_args() #ipykernel_launcher.py: error: unrecognized arguments. Replaced by the line below.

    args, unknown = parser.parse_known_args()  #revised. Solution: 2nd answer in 

    #https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

    path = os.path.join(MODELS_PATH, args.model)

    if not os.path.exists(path):

        os.makedirs(path)

    if args.model not in list(MODELS_PRETRAINED) + list(MODELS_FINETUNED):

        raise AttributeError("Model should be selected in the list: " + 

            ", ".join(list(MODELS_PRETRAINED) + list(MODELS_FINETUNED))

        )

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])

    model = AutoModel.from_pretrained(MODELS[args.model])

    model.save_pretrained(path)

    tokenizer.save_pretrained(path)

    if args.model in MODELS_FINETUNED.keys(): # Build the SentenceTransformer directly

        word_embedding_model = models.BERT(

            path,

            max_seq_length=args.max_seq_length,

            do_lower_case=args.do_lower_case

        )

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),

                               pooling_mode_mean_tokens=True,

                               pooling_mode_cls_token=False,

                               pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        rmtree(path)

        model.save(path)

    print(f'Model {args.model} available in', path)



#scibert-nli model downloaded
df = pd.read_csv("../input/biorxiv_clean.csv")

print(df.shape)

df.head(3)
# Download datasets to train NLI model, also runs once for all #如果用训练好的pretrained model，就不用这段代码了

# modified from https://github.com/UKPLab/sentence-transformers/blob/master/examples/datasets/get_data.py



######## Run once for all

folder_path = os.getcwd()

print('Beginning download of datasets')



datasets = ['AllNLI.zip', 'stsbenchmark.zip', 'wikipedia-sections-triplets.zip'] #下载这些dataset    ###这些data里分别都是什么呢?可以从名字大概推测但。。。

server = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/"



#download dataset

for dataset in datasets:

    print("Download", dataset)

    url = server+dataset

    dataset_path = os.path.join(folder_path, dataset)

    urllib.request.urlretrieve(url, dataset_path)



    print("Extract", dataset)

    with zipfile.ZipFile(dataset_path, "r") as zip_ref:

        zip_ref.extractall(folder_path)

    os.remove(dataset_path)

    #break 只download第一个dataset



print("All datasets downloaded and extracted")
#one model a time

batch_size = 16

nli_reader = NLIDataReader('AllNLI')

sts_reader = STSDataReader('stsbenchmark') 

train_num_labels = nli_reader.get_num_labels() 



dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model) #test data

dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size) #有放回抽样

evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)



# evaluate model

model.evaluate(evaluator) 
##### need to run "ask_question" function for each different model. won't take long

def cache_corpus(df): #original version

    corpus = [a for a in df['abstract'] if type(a) == str and a != "Unknown"]

    print('Corpus size', len(corpus))



    with open("corpus.pkl", 'wb') as file:

        pickle.dump(corpus, file)

    return corpus



def ask_question(query, model, corpus, corpus_embed, top_k=5): #For this query(question), using this model and abstracts in the corpus, list the top 5 corpus, 

    """

    Adapted from https://www.kaggle.com/dattaraj/risks-of-covid-19-ai-driven-q-a

    """

    queries = [query]

    query_embeds = model.encode(queries, show_progress_bar=False)

    for query, query_embed in zip(queries, query_embeds):

        # cosine similarity here, can tune

        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0] #计算distance between query and corpus_embed(????哪来的)

        distances = zip(range(len(distances)), distances) #dict

        distances = sorted(distances, key=lambda x: x[1]) #排序

        results = []

        for count, (idx, distance) in enumerate(distances[0:top_k]):

            results.append([count + 1, corpus[idx].strip(), round(1 - distance, 4)])

    return results





def show_answers(results):

    table = prettytable.PrettyTable(['Rank', 'Abstract', 'Score']) #print this table

    for res in results:

        rank = res[0]

        text = textwrap.fill(res[1], width=75) + '\n\n'

        score = res[2]

        table.add_row([rank, text, score])

    print('\n')

    print(str(table))

    print('\n')
corpus = cache_corpus(df)    #####this also takes a long time. 

embeddings = model.encode(corpus, show_progress_bar=False) #吧corpus整个embed成vector

with open("embeddings.pkl", 'wb') as file:

    pickle.dump(embeddings, file)
query = "What is known about transmission, incubation, and environmental stability?" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "What do we know about COVID-19 risk factors?" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Smoking, pre-existing pulmonary disease" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities." 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Neonates and pregnant women" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Susceptibility of populations" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Animal host(s) and any evidence of continued spill-over to humans" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Socioeconomic and behavioral risk factors for this spill-over" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Sustainable risk reduction strategies" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "What has been published about ethical and social science considerations?" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Efforts to support sustained education, access, and capacity building in the area of ethics" 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question
query = "Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media." 

results = ask_question(query, model, corpus, embeddings)

show_answers(results)  #result is 5 abstract with highest score according to the question