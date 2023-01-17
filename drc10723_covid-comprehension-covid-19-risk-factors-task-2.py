%%capture

!pip install -q sentence-transformers==0.2.5.1

!pip install -q transformers==2.5.1

!python -m spacy download en_core_web_sm
import os

import pickle

import logging

import json

import scipy

import spacy



import numpy as np 

import pandas as pd

from tqdm import tqdm

from collections import OrderedDict

from sentence_transformers import SentenceTransformer

from transformers import pipeline
DATA_PATH = '/kaggle/input/CORD-19-research-challenge/'

BIORXIV_PATH = os.path.join(DATA_PATH, 'biorxiv_medrxiv/biorxiv_medrxiv/')

COMM_USE_PATH = os.path.join(DATA_PATH, 'comm_use_subset/comm_use_subset/')

NONCOMM_USE_PATH = os.path.join(DATA_PATH, 'noncomm_use_subset/noncomm_use_subset/')

CUSTOM_PATH = os.path.join(DATA_PATH, 'custom_license/custom_license/')

METADATA_PATH = os.path.join(DATA_PATH, 'metadata.csv')

CACHE_PATH = '/cache'

if not os.path.exists(CACHE_PATH):

    os.mkdir(CACHE_PATH)

MODELS_PATH = '/models'

if not os.path.exists(MODELS_PATH):

    os.mkdir(MODELS_PATH)

RANK_USING = 'abstract'     # rank using: abstract(default) / title / text

MODEL_NAME = 'scibert-nli'  # model: scibert-nli / biobert-nli / covidbert-nli

COMPREHENSION_MODEL = "graviraja/covidbert_squad"     # model used for comprehension

COMPREHENSION_TOKENIZER = "graviraja/covidbert_squad"                 # tokenizer for comprehension

use_gpu = -1                                                        # use the gpu

CORPUS_PATH = os.path.join(CACHE_PATH, 'corpus.pkl')                 # processed corpus path

MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)                  # path to the saved model

EMBEDDINGS_PATH = os.path.join(

    CACHE_PATH, f'{MODEL_NAME}-{RANK_USING}-embeddings.pkl')         # path to save the computed embeddings
def get_all_files(dir_name):

    """

    Get all the files in the given directory and all it's sub-directories

    Parameters

    ----------

    dir_name: str

        A directory path

    Returns

    -------

    all_files: list

        A list containing all the files in the given directory

    """

    list_of_file = os.listdir(dir_name)

    all_files = list()



    for entry in list_of_file:

        full_path = os.path.join(dir_name, entry)

        # If entry is a directory then get the list of files in this directory

        if os.path.isdir(full_path):

            all_files = all_files + get_all_files(full_path)

        else:

            if full_path.endswith('json'):

                all_files.append(full_path)



    return all_files





def load_files(path):

    """

    Load all the files in the given path

    Parameters

    ----------

    path: str

        path to a directory containing files

    Returns

    -------

    all_files: list

        A list containing the loaded files in json

    """

    filenames = get_all_files(path)

    print(f"Number of files retrived from {path}: {len(filenames)}")

    all_files = []



    for each_file in tqdm(filenames, desc="Reading files: "):

        with open(each_file, 'rb') as f:

            file = json.load(f)

        all_files.append(file)

    return all_files





def format_name(author):

    """

    Formats the author name into a standard format

    """

    middle_name = " ".join(author['middle'])



    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    """

    Formats the affiliation details by location and institution

    """

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))



    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)





def format_authors(authors, with_affiliation=False):

    """

    Formats each author name with optional affiliation details

    """

    name_ls = []



    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)



    return ", ".join(name_ls)





def format_body(body_text):

    """

    Formats the sections of the paper into a single body

    """

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}



    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += text

        body += "\n\n"



    return body







def generate_clean_df(all_files, metadata, mode="abstract"):

    """

    Formats each file and links with it's metada.

    Parameters

    ----------

    all_files: list

        A list of json files

    metadata: pandas.DataFrame

        A pandas dataframe containing the metadata

    mode: str

        A string indicating the mode used for ranking

    Returns

    -------

    clean_df: pandas.DataFrame

        A pandas dataframe containing the cleaned data

    """

    cleaned_files = []

    metadata_not_found = 0

    mode_not_found = 0



    for file in tqdm(all_files):

        mode_data = ""

        if mode == "abstract":

            try:

                if 'abstract' in file.keys():

                    mode_data = format_body(file['abstract'])

                elif 'abstract' in file['metadata'].keys():

                    mode_data = format_body(file['metadat']['abstract'])

                else:

                    mode_not_found += 1

                    continue

            except Exception as e:

                mode_not_found += 1

                continue

        elif mode == "text":

            mode_data = format_body(file['body_text'])

        elif mode == "title":

            mode_data = file['metadata']['title']

        if mode_data == "":

            mode_not_found += 1

            continue



        id_row = metadata.loc[metadata['sha'] == file['paper_id']]

        row = id_row

        metadata_found = True

        if id_row.empty:

            title_row = metadata.loc[metadata['title'] == file['metadata']['title']]

            if title_row.empty:

                metadata_not_found += 1

                metadata_found = False

            else:

                row = title_row

        if metadata_found:

            row = row.iloc[0]

            cord_uid = row['cord_uid']

            publish_time = row['publish_time']

            url = row['url']

            source = row['source_x']

            license = row['license']

        else:

            cord_uid = ''

            publish_time = ''

            url = ''

            source = ''

            license = ''



        try:

            paper_id = file['paper_id']

        except Exception as e:

            paper_id = ''



        try:

            title = file['metadata']['title']

        except Exception as e:

            title = 'Title not found'



        try:

            authors = format_authors(file['metadata']['authors'])

        except Exception as e:

            authors = 'Authors not found'



        try:

            affiliation = format_authors(

                file['metadata']['authors'], with_affiliation=True)

        except Exception as e:

            affiliation = ''



        try:

            if 'abstract' in file.keys():

                abstract = format_body(file['abstract'])

            elif 'abstract' in file['metadat'].keys():

                abstract = format_body(file['metadat']['abstract'])

            else:

                abstract = ''

        except Exception as e:

            abstract = ''



        try:

            body = format_body(file['body_text'])

        except Exception as e:

            body = ''



        features = [

            paper_id,

            cord_uid,

            title,

            publish_time,

            authors,

            affiliation,

            abstract,

            body,

            url,

            source,

            license

        ]



        cleaned_files.append(features)



    print(f"Metadata not found for {metadata_not_found} files")

    print(f"{mode} is null for {mode_not_found} files")

    print(f"considered {len(cleaned_files)} files")

    col_names = ['paper_id', 'cord_uid', 'title', 'publish_time', 'authors',

                 'affiliations', 'abstract', 'text', 'url', 'source', 'license']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()



    return clean_df



def generate_clean_csv(datapath, metadata_path, source, datafolder, mode):

    """

    Formats each of the file in datapath and link it's metadata in metadata_path.

    Parameters

    ----------

    datapath: str

        A string containing the path to raw files

    metadata_path: str

        A string containing the path to metadata.csv file

    source: str

        A string indicating the source which is being processed

    datafolder: str

        A string containing the path to save the cleaned data

    mode: str

        A string indicating the mode of ranking

    Returns

    -------

    clean_df: pandas.DataFrame

        A pandas dataframe containing the cleaned data

    """

    files = load_files(datapath)

    metadata = pd.read_csv(metadata_path)

    clean_df = generate_clean_df(files, metadata, mode)

    print(f"Saving the cleaned data into file: {datafolder}/clean_{source}.csv")

    clean_df.to_csv(f'{datafolder}/clean_{source}.csv', index=False)

    return clean_df
def cache_corpus(mode):

    """

    For each datapath, clean the data and cache the cleaned data.

    Parameters

    ----------

    mode: str

        A string indicating the mode of ranking. (abstract / title / text)

    """

    biorxiv_df = generate_clean_csv(

        BIORXIV_PATH, METADATA_PATH, 'biorxiv', CACHE_PATH, mode)

    comm_use_df = generate_clean_csv(

        COMM_USE_PATH, METADATA_PATH, 'comm_use', CACHE_PATH, mode)

    noncomm_use_df = generate_clean_csv(

        NONCOMM_USE_PATH, METADATA_PATH, 'noncomm_use', CACHE_PATH, mode)

    custom_df = generate_clean_csv(

        CUSTOM_PATH, METADATA_PATH, 'custom', CACHE_PATH, mode)



    corpus = pd.concat(

        [biorxiv_df, comm_use_df, noncomm_use_df, custom_df], ignore_index=True)

    print("Creating corpus by combining: biorxiv, comm_use, noncomm_use, custom data")

    with open(CORPUS_PATH, 'wb') as file:

        pickle.dump(corpus, file)

    return corpus

if not os.path.exists(CORPUS_PATH):

    print(

        "If the RANK_USING mode is modified means delete the cache and recreate it.")

    print("Caching the corpus for future use...")

    corpus = cache_corpus(RANK_USING)

else:

    print(f"Loading the corpus from {CORPUS_PATH}...")

    with open(CORPUS_PATH, 'rb') as corpus_pt:

        corpus = pickle.load(corpus_pt)
import os

from transformers import AutoTokenizer, AutoModel

from sentence_transformers import models, SentenceTransformer



MODELS = {

    'scibert-nli': 'gsarti/scibert-nli',

    'biobert-nli': 'gsarti/biobert-nli',

    'covidbert-nli': 'gsarti/covidbert-nli'

}



def load_sentence_transformers_model(model_name, path):

    if model_name not in MODELS:

        raise AttributeError("Model should be selected in the list: " +

                             ", ".join(list(MODELS))

                             )

    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])

    model = AutoModel.from_pretrained(MODELS[model_name])

    print(path)

    model.save_pretrained(path)

    tokenizer.save_pretrained(path)

    # Build the SentenceTransformer directly

    word_embedding_model = models.BERT(

        path,

        max_seq_length=128,

        do_lower_case=True

    )

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),

                                   pooling_mode_mean_tokens=True,

                                   pooling_mode_cls_token=False,

                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(

        modules=[word_embedding_model, pooling_model])

    

    return model
model = load_sentence_transformers_model(MODEL_NAME, MODELS_PATH)



if RANK_USING == "abstract":

    rank_corpus = corpus['abstract'].values

elif RANK_USING == "text":

    rank_corpus = corpus['text'].values

elif RANK_USING == "title":

    rank_corpus = corpus['title'].values

else:

    raise AttributeError(

        "Ranking should be with abstract, text (or) title are only supported")
if not os.path.exists(EMBEDDINGS_PATH):

    print("Computing and caching model embeddings for future use...")

    embeddings = model.encode(rank_corpus, show_progress_bar=True)

    with open(EMBEDDINGS_PATH, 'wb') as file:

        pickle.dump(embeddings, file)

else:

    print(f"Loading model embeddings from {EMBEDDINGS_PATH}...")

    with open(EMBEDDINGS_PATH, 'rb') as file:

        embeddings = pickle.load(file)

# model used for comprehension

comprehension_model = pipeline("question-answering", model=COMPREHENSION_MODEL,

                               tokenizer=COMPREHENSION_TOKENIZER, device=use_gpu)
def rank_with_bert(query, model, corpus, corpus_embed, top_k=5):

    """

    Document Ranking

    Converts the query into an embedding using model. Uses the query embedding and

    corpus embeddings to calculate the cosine similarity. Sorted the results according

    to cosine scores. For the top_k documents, maps the relevant metadata and returns

    the results.

    Parameters

    ----------

    query: str

        A query string

    model:

        Model used for creating embeddings. Default is scibert-nli

    corpus:

        Cleaned corpus

    corpus_embed:

        Embeddings which will be used for ranking

    top_k: int (default = 5)

        Number of top documents to retrieve

    Returns

    -------

    results: list

        Updates each document in the list with document ranking results

    """

    queries = [query]

    query_embeds = model.encode(queries, show_progress_bar=False)

    for query, query_embed in zip(queries, query_embeds):

        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]

        distances = zip(range(len(distances)), distances)

        distances = sorted(distances, key=lambda x: x[1])

        results = []

        for count, (idx, distance) in enumerate(distances[0:top_k]):

            doc = corpus.iloc[idx]

            paras = doc['text'].split('\n\n')

            paras = [para for para in paras if len(para) > 0]

            result = {}

            result['document_rank'] = count + 1

            result['document_score'] = round(1 - distance, 4)

            result['paper_id'] = doc['paper_id']

            result['cord_uid'] = doc['cord_uid']

            result['title'] = doc['title']

            result['publish_time'] = doc['publish_time']

            result['authors'] = doc['authors']

            result['affiliations'] = doc['affiliations']

            result['abstract'] = doc['abstract']

            result['paragraphs'] = paras

            result['url'] = doc['url']

            result['source'] = doc['source']

            result['license'] = doc['license']

            results.append(result)

    return results
def paragraph_ranking(query, model, documents, top_k=5):

    """

    Paragraph Ranking

    Converts the query into an embedding using model. Converts the paragraphs in the

    document into embeddings using the model. Uses the query embedding and

    paragraph embeddings to calculate the cosine similarity. Sorted the results according

    to cosine scores. For the top_k paragraphs, maps the relevant metadata and returns

    the results.

    Parameters

    ----------

    query: str

        A query string

    model:

        A model to convert the paragraphs and query into embeddings. Default is scibert-nli

    documents:

        A list containing the documents

    top_k: int (default = 5)

        Number of top paragraphs to retrieve from each document

    Returns

    -------

    documents: list

        Updates each document in the list with paragraph ranking results

    """

    for each_doc in documents:

        paras = each_doc["paragraphs"]

        para_embeds = model.encode(paras, show_progress_bar=False)

        query_embed = model.encode([query], show_progress_bar=False)

        distances = scipy.spatial.distance.cdist(query_embed, para_embeds, "cosine")[0]

        paragraph_ranking = []

        for para_idx, para_score in enumerate(distances):

            result = {}

            result['paragraph_id'] = para_idx

            result['paragraph_score'] = round(1 - para_score, 4)

            paragraph_ranking.append(result)

        sorted_paras = sorted(paragraph_ranking, key=lambda x: x['paragraph_score'], reverse=True)

        paragraph_ranking_results = sorted_paras[0:top_k]

        each_doc["paragraph_ranking"] = paragraph_ranking_results

    return documents

def show_ranking_results(results):

    """

    Prints the document ranking results

    Parameters

    ----------

    results: list

        List of top retrieved documents

    """

    result = {"title": [], "doc_rank": [], "doc_score": [], "abstract": []}

    for r in results:

        result['title'].append(r['title'])

        result['doc_rank'].append(r['document_rank'])

        result['doc_score'].append(r['document_score'])

        result['abstract'].append(r['abstract'])

    df = pd.DataFrame(result)

    print(df)
spacy_nlp = spacy.load('en_core_web_sm')



N_BEST_PER_PASSAGE = 1





def get_full_sentence(para_text, start_index, end_index):

    """

    Get surrounding sentence

    Parameters

    ----------

    para_text: str

        Paragraph text in form of string

    start_index: int

        start index in the original text

    end_index: int

        end index in the original text

    Returns

    -------

        surrounding sentence for original text. it's start index and end index.

    """

    sent_start = 0

    sent_end = len(para_text)

    for sent in spacy_nlp(para_text).sents:

        if (sent.start_char <= start_index) and (sent.end_char >= start_index):

            sent_start = sent.start_char

        if (sent.start_char <= end_index) and (sent.end_char >= end_index):

            sent_end = sent.end_char

    sentence = para_text[sent_start:sent_end + 1]

    return sentence, sent_start, sent_end





def result_on_one_document(model, question, document, top_k=5):

    """

    Comprehension on one document

    For the top retrieved paragraphs in the document, answer will be comprehended

    on each paragraph using the model.

    Parameters

    ----------

    model: Any HuggingsFace Model.

        Model used for comprehension

    question: str

        A question string

    document: dict

        A document containing paragraphs

    Returns

    -------

    document: dict

        Updates the given document dictionary with comprehension results

    """

    paragraphs = document["paragraphs"]

    paragraph_rank_results = document["paragraph_ranking"]

    answers = []

    for idx, para in enumerate(paragraph_rank_results):

        para_idx = para["paragraph_id"]

        query = {"context": paragraphs[para_idx], "question": question}

        pred = model(query, topk=N_BEST_PER_PASSAGE)

        # assemble and format all answers

        # for pred in predictions:

        if pred["answer"] and round(pred["score"], 4) > 0:

            sent, start, end = get_full_sentence(query['context'], pred["start"], pred["end"])

            ans_result = OrderedDict()

            ans_result['title'] =document['title']

            ans_result['context'] = paragraphs[para_idx]

            ans_result['relevant sentence'] = sent

            ans_result['compre_score'] = round(pred["score"], 4)

            #ans_result['article_url'] = document['url']

            answers.append(ans_result)

    return answers





def comprehend_with_bert(model, question, documents):

    """

    Document Comprehension

    For the top retrieved paragraphs in each document, answer will be comprehended

    on each paragraph using the model.

    Parameters

    ----------

    model: Any HuggingsFace Model.

        Model used for comprehension

    question: str

        A question string

    documents: list

        A list of documents containing paragraphs

    Returns

    -------

    comprehend_results: list

        Updates each document in the list with comprehension results

    """

    comprehend_results = []

    for document in documents:

        result = result_on_one_document(model, question, document)

        comprehend_results.extend(result)

    return comprehend_results





def show_comprehension_results(results):

    """

    Prints the comprehension results

    Parameters

    ----------

    results: list

        List of comprehended documents

    """

    result = {"answer": [], "context": [], "title": [], "score": [], "doc_rank": [], "doc_score": []}

    for r in results:

        result['answer'].append(r['answer'])

        result['context'].append(r['context'])

        result['title'].append(r['title'])

        result['score'].append(r['probability'])

        result['doc_rank'].append(r['document_rank'])

        result['doc_score'].append(r['document_score'])

    df = pd.DataFrame(result)

    print(df)
def query_covid(query):

    # ranking the documents using pre-calculated embeddings and query

    document_rank_results = rank_with_bert(query, model, corpus, embeddings)



    # ranking the paragraphs of the top retrieved documents

    paragraph_rank_results = paragraph_ranking(query, model, document_rank_results)



    # comprehending the top paragraphs of the retrieved documents for finding answer

    comprehend_results = comprehend_with_bert(comprehension_model, query, paragraph_rank_results)

     # sort answers by their `probability` and select top-k

    comprehend_results = sorted(comprehend_results, key=lambda k: k["compre_score"], reverse=True)

    #answers = answers[:top_k]

    df = pd.DataFrame(comprehend_results)

    df = df.drop('compre_score', axis=1)

    return df
%%capture

result = query_covid("what are common symptoms observed in covid 19 patients ?")
from IPython.display import display, HTML



display(HTML(result.to_html()))
%%capture

result = query_covid("How does covid 19 affect smokers")
display(HTML(result.to_html()))
%%capture

result = query_covid("What is fatality rate for covid 19 patients with smoking ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("How does covid 19 affect pre-existing pulmonary disease ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("how does co-existing viral infections affect transmission ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("on what rate of covid 19 transmission depends?")
display(HTML(result.to_html()))
%%capture

result = query_covid("What do we know about covid 19 in pregant womens ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("how does socio-economic correlate with covid 19 spread ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("Is total lockdown effective for stopping covid 19 transmission ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("what are environmental factors in covid 19 transmission ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("what are modes of covid 19 transmission ?")
display(HTML(result.to_html()))
%%capture

result = query_covid(" What is serverity of covid 19  and who are high risk patients ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("does covid 19 transmission vary based on sex or race ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("which public health mitigation can be effective for controlling covid 19 spread ?")
display(HTML(result.to_html()))
%%capture

result = query_covid("how does covid 19 varies via country or regions ?")
display(HTML(result.to_html()))