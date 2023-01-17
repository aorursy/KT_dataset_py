!pip install sentence-transformers
import os

import re  # For preprocessing

import en_core_web_sm

from difflib import SequenceMatcher

import pandas as pd

import numpy as np

import pickle

from time import time  # To time our operations

import glob

import json

import zipfile

from tqdm import tqdm

import multiprocessing

import scipy



import torch

from transformers import BertTokenizer

from transformers import BertForQuestionAnswering

from sentence_transformers import SentenceTransformer

from typing import List, Dict

from sklearn.metrics.pairwise import cosine_similarity #for cosine similarity

from gensim.models.phrases import Phrases, Phraser #For create relevant phrases

from gensim.models import Word2Vec #Our model type

import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
root_path = '/kaggle/input/CORD-19-research-challenge'



metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str,

    'doi': str

})

meta_df.head()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            if 'abstract' in content:

                for entry in content['abstract']:

                    self.abstract.append(entry['text'])

            else:

                self.abstract.append('Not provided.')

            # Body text

            if 'body_text' in content:

                for entry in content['body_text']:

                    self.body_text.append(entry['text'])

            else:

                self.body_text.append('Not provided.')

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)





    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def get_date_dt(all_json, meta_df):

    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [],

             'abstract_summary': []}

    for idx, entry in tqdm(enumerate(all_json), desc="Parsing the articles Json's content", total=len(all_json)):

        content = FileReader(entry)



        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

        # no metadata, skip this paper

        if len(meta_data) == 0:

            continue



        dict_['paper_id'].append(content.paper_id)

        dict_['abstract'].append(content.abstract)

        dict_['body_text'].append(content.body_text)

        

        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]



        try:

            # if more than one author

            authors = meta_data['authors'].values[0].split(';')

            if len(authors) > 2:

                # more than 2 authors, may be problem when plotting, so take first 2 append with ...

                dict_['authors'].append(". ".join(authors[:2]) + "...")

            else:

                # authors will fit in plot

                dict_['authors'].append(". ".join(authors))

        except Exception as e:

            # if only one author - or Null valie

            dict_['authors'].append(meta_data['authors'].values[0])



        # add the title information, add breaks when needed

        dict_['title'].append(meta_data['title'].values[0])



        # add the journal information

        dict_['journal'].append(meta_data['journal'].values[0])

    return pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal'])
def Initialization_Word2Vec_model():

    w2v_model = Word2Vec(min_count=50,

                     window=10,

                     size=200,

                     sample=6e-5,

                     alpha=0.02,

                     min_alpha=0.0003,

                     negative=20,

                     workers=multiprocessing.cpu_count() -1)

    return w2v_model

def create_sentences(df, col_name):

    """Build sentences to Word2Vec model """

    t = time()

    sent = [row.split() for row in df[col_name]]

    phrases = Phrases(sent, min_count=50, progress_per=100, max_vocab_size=1000000)

    bigram = Phraser(phrases)



    print('Time to create sentences: {} mins'.format(round((time() - t) / 60, 2)))

    return bigram, bigram[sent]



def Build_Word2Vec_vocab(w2v_model, sentences,update=True):

    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000, update=update)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    return w2v_model



def Train_Word2Vec_model(w2v_model, sentences):

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    return w2v_model



def cleaning(spacy_doc):

    """Lemmatizes and removes stopwords

    doc needs to be a spacy Doc object """



    txt = [t.lemma_ for t in spacy_doc if

            t.dep_ not in ['prep', 'punct', 'det'] and

            len(t.text.strip()) > 2 and

            t.lemma_ != "-PRON-" and

            not t.is_stop]

    if len(txt) > 2:

        return ' '.join(txt)

    else:

        return "no text"



def word2vec_preprocessing(df, column, offline=True) -> pd.DataFrame:

    """ Prepare column from pd.DataFrame with text to Word2Vec model """

    url_pattern = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"

    data = df.drop_duplicates(column)

    data = data[

        data[column].apply(lambda x: len(x) > 0) &

        data[column].apply(lambda x: len(x.split()) > 3)

        ]

    data.reset_index(inplace=True, drop=True)

    data[column] = data[column].map(lambda x: re.sub(url_pattern, ' ', str(x)))

    brief_cleaning = (re.sub(r"[^a-zA-Z']+", ' ', str(row)).lower() for row in data[column])

    nlp = en_core_web_sm.load(disable=['ner', 'parser', 'tagger'])  # disabling Named Entity Recognition for speed

    t = time()

    txt = []

    if offline:

        paper_ids = data['paper_id']

        index = []

        for idx, doc in enumerate(nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1, n_process=5)):

            txt.append(cleaning(doc))

            index.append(paper_ids[idx]) #TODO verify its ok

        print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    else:

        for doc in nlp.pipe(brief_cleaning):

            txt.append(cleaning(doc))



    df_clean = data

    print([t for t in txt if t is None])

    df_clean[f'clean_{column}'] = txt

    return df_clean

def preprocessing(corpus: List, is_offline: bool):

    dt_corpus = pd.DataFrame({'clean': corpus})

    procesed = word2vec_preprocessing(dt_corpus, 'clean', offline=is_offline)

    return procesed['clean_clean'].to_list()



def get_relevant_articles(query,  articles_abstract):

    queries: List[str] = preprocessing([query], is_offline=False)

    articles_abstract.score = articles_abstract.score.astype(float) # to make the scores as float in order to not lose precision

    for q in queries:

        query_key_words: List[str] = get_unk_kw(phraser, q)

        embedded_query = embed_sentence(w2v_model, query_key_words)

        for idx, abstract in tqdm(articles_abstract.iterrows(), desc="Iterate over all articles abstract", total=len(articles_abstract)):

            abs_score = get_abstract_score(w2v_model, embedded_query, abstract['clean_abstract'])

            articles_abstract.at[idx, 'score'] = abs_score

    scored_articles_abstract = articles_abstract.sort_values('score', ascending=False)

    return scored_articles_abstract
data_dt = get_date_dt(all_json, meta_df)
to_train = False

if to_train:

    w2v_model = Initialization_Word2Vec_model()

    df_clean = word2vec_preprocessing(data_dt, 'body_text')

    phraser, sentences = create_sentences(df_clean, 'clean_body_text')

    update_w2v_vocab = len(w2v_model.wv.vocab) != 0

    w2v_model = Build_Word2Vec_vocab(w2v_model, sentences, update_w2v_vocab)

    w2v_model = Train_Word2Vec_model(w2v_model, sentences)

    # w2v_model.save('saved_model/w2v_model_on_all_abstract_full_text.w2v')

    w2v_model.init_sims(replace=True)

else:

    with open("/kaggle/input/w2v-model/saved_model/cleaned_to_w2v_all_document.pkl", 'rb') as f:

        df_clean = pickle.load(f)

    with open("/kaggle/input/w2v-model/phraser.pkl", 'rb') as f:

        phraser = pickle.load(f)

    w2v_model = Word2Vec.load('/kaggle/input/w2v-model/saved_model/w2v_model_on_all_abstract_full_text.w2v')
print("most similar words to kaletra:")

for s_w in w2v_model.wv.most_similar(positive=[ 'kaletra']):

    print(s_w)

print('#'*100)

print("most similar words to bat:")

for s_w in w2v_model.wv.most_similar(positive=['bat']):

    print(s_w)

print('#'*100)

print("most similar words to dead and people:")

for s_w in w2v_model.wv.most_similar(positive=['dead', 'people']):

    print(s_w)
ALPHA = 0.5



def embed_sentence(model, tokens: List[str]) -> List:

    res = []

    for t in tokens:

        try:

            vec = model.wv.word_vec(t, use_norm=False)

            res.append(vec)

        except KeyError:

            # logging.debug(f'Unidentified word while embedding:{t}')

            continue

    return res





def get_all_abstracts(phraser, cleaned_df):

    column = 'abstract'

    df = word2vec_preprocessing(cleaned_df, column, offline=True)

    df[f'clean_{column}'] = df[f'clean_{column}'].map(lambda x: get_unk_kw(phraser, x))

    df['score'] = [0] * len(df)

    return df





def get_unk_kw(phraser, query):

    if type(query) == str:

        return list(set(phraser[query.split()]))

    else:

        return []





def preprocessing(corpus: List, is_offline: bool):

    dt_corpus = pd.DataFrame({'clean': corpus})

    procesed = word2vec_preprocessing(dt_corpus, 'clean', offline=is_offline)

    return procesed['clean_clean'].to_list()





def get_abstract_score(model: Word2Vec, embedded_query:List, abstract: List[str]) -> float:

    f_score = 0

    valid_scores_sum = 1  # so we wouldn't divide by zero

    if type(abstract) != list or len(abstract) == 0:

        return f_score

    embedded_abstract = embed_sentence(model, abstract)

    for q_t in embedded_query:

        scores = model.wv.cosine_similarities(q_t, embedded_abstract)

        valid_scores = [s for s in scores if s > ALPHA]

        valid_scores_sum += len(valid_scores)

        f_score += np.sum(valid_scores)

    norm_score = f_score / valid_scores_sum # to normalize by the number of tokens that was counted

    return norm_score

# articles_abstract: pd.DataFrame = get_all_abstracts(phraser, all_data)
print(w2v_model.wv.most_similar(positive=['medicine', 'covid'], negative=['sars']))

print(w2v_model.wv.most_similar(positive=['dead', 'covid','israel']))

print(w2v_model.wv.similarity('kaletra', 'covid'))

print(w2v_model.wv.similarity('kaletra', 'sars'))



with open('/kaggle/input/w2v-model/parsed_abstract_ran_offline.pk', 'rb') as f: 

    articles_abstract = pickle.load(f)

articles_abstract['body_text'].head()
encoder = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
encoded_articles_abstract = encoder.encode(articles_abstract['abstract'].tolist())

articles_abstract['encoded_articles_abstract'] = encoded_articles_abstract    
def query_questions(query, articles_abstract):    

    encoded_query = encoder.encode([query])

    articles_abstract['distances'] = scipy.spatial.distance.cdist(encoded_query, articles_abstract['encoded_articles_abstract'].tolist(), "cosine")[0]

    articles_abstract = articles_abstract.sort_values('distances').reset_index()[:70]

    

    articles_abstract['sentence_list'] = [body.split(". ") for body in articles_abstract['body_text'].to_list()] 

    paragraphs = []

    for index, ra in articles_abstract.iterrows():

        para_to_add = [". ".join(ra['sentence_list'][n:n+7]) for n in range(0, len(ra['sentence_list']), 7)]        

        para_to_add.append(ra['abstract'])

        paragraphs.append(para_to_add)

    articles_abstract['paragraphs'] = paragraphs

    answers = answer_question(query, articles_abstract)

    return answers
from transformers import AutoTokenizer

from transformers import AutoModelForQuestionAnswering

from transformers import pipeline



def get_QA_bert_model():

    torch_device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")

    model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")

    return pipeline('question-answering', model=model, tokenizer=tokenizer,device=torch_device)
nlp_qa = get_QA_bert_model()
def answer_question(question: str, context_list):

    # anser question given question and context

    answers =[]

    all_para = [item for sublist in context_list['paragraphs'].to_list() for item in sublist] 

    print(f"paragraph to scan: {len(all_para)}")

    for _, article in context_list.iterrows():

        for context in article['paragraphs']:

            if len(context) < 10:

                continue

            with torch.no_grad():

                answer = nlp_qa(question=question, context=context)

            answer['paragraph'] = context

            answer['paper_id'] = article['paper_id']

            answers.append(answer)            

    df = pd.DataFrame(answers)

    df = df.sort_values(by='score', ascending=False)

    return df



# def answer_question(question: str, context_list):

#     # anser question given question and context

#     torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     answers =[]

#     all_para = [item for sublist in context_list['paragraphs'].to_list() for item in sublist] 

#     print(f"paragraph to scan: {len(all_para)}")

#     i=0

#     for _, article in context_list.iterrows():

#         for context in article['paragraphs']:

#             if len(context) < 10:

#                 continue

#             i = i+1

#             if i % 100 == 0:

#                 print(i)

                

#             encoded_dict = tokenizer.encode_plus(

#                                 question, context,

#                                 add_special_tokens = True,

#                                 max_length = 500,

#                                 pad_to_max_length = True,

#                                 return_tensors = 'pt'

#                            )



#             input_ids = encoded_dict['input_ids'].to(torch_device)

#             token_type_ids = encoded_dict['token_type_ids'].to(torch_device)

#             with torch.no_grad():  

#                 start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)

#             all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

#             start_index = torch.argmax(start_scores)

#             end_index = torch.argmax(end_scores)

# #             print("total score: ", start_scores[0][start_index].float() + end_scores[0][end_index].float())

#             answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])

#             answer = answer.replace('[CLS]', '').replace('<pad>', '').replace('[SEP]', '')

#             if len(answer.strip()) > 6 and similar(question.lower().strip(), answer.lower().strip()) < 0.8 and question.lower().strip() not in answer.lower().strip():

#                 answers.append({'answer': answer, 'paragraph': context, 'paper_id': article['paper_id'], 'total_score':start_scores[0][start_index].item()+end_scores[0][end_index].item() })

#     return pd.DataFrame(answers)
answer1 = query_questions("Are there geographic variations in the rate of COVID-19 spread?", articles_abstract)

answer1.to_csv("/kaggle/working/1_Are there geographic variations in the rate of COVID-19 spread.csv")

answer1.head()
answer2 = query_questions("What works have been done on infection spreading?", articles_abstract)

answer2.to_csv("/kaggle/working/2_What works have been done on infection spreading.csv")

answer2.head()
answer3 =  query_questions("Are there geographic variations in the mortality rate of COVID-19?", articles_abstract)

answer3.to_csv("/kaggle/working/3_Are there geographic variations in the mortality rate of COVID.csv")

answer3.head()
answer4 =  query_questions("Is there any evidence to suggest geographic based virus mutations?", articles_abstract)

answer4.to_csv("/kaggle/working/4_Is there any evidence to suggest geographic based virus mutations.csv")

answer4.head()