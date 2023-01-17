import os

import re

import gc

import sys

import math

import glob

import json

import spacy

import pandas

import random

import warnings

import numpy as np

import cufflinks as cf



from spacy import displacy

from tqdm.notebook import tqdm

from collections import Counter 

from collections import defaultdict

from nltk.corpus import stopwords

from IPython.core.display import HTML, Markdown



# filter warnings

warnings.filterwarnings('ignore')



# defining some options for displacy

colors = {"KP": "#2fbbab"}

options = {"ents": ["KP"], "colors": colors}



# defining some options for pandas

pandas.set_option('display.max_rows', 5)

pandas.set_option('display.max_columns', None)



# defining some options for cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
v7_path = '/kaggle/input/CORD-19-research-challenge'

df = pandas.read_csv(f'{v7_path}/metadata.csv', 

    usecols=['cord_uid', 'sha', 'pmcid', 'title', 'abstract', 'publish_time', 'authors'],

    dtype={'cord_uid' : str, 'sha': str, 'pmcid': str, 'title': str, 'abstract': str, 'authors': str},

    parse_dates=['publish_time'], 

    keep_default_na=False

)
def code_we_used_on_our_server() :

    pdf_json_dict = dict()

    for filename in tqdm(glob.glob(f'{v7_path}/**/pdf_json/*.json', recursive=True), leave = False): 

        pdf_json_dict[json.load(open(filename, 'rb'))['paper_id']] = filename

    print(len(pdf_json_dict), "papers from PDF parsing")



    xml_json_dict = dict()

    for filename in tqdm(glob.glob(f'{v7_path}/**/pmc_json/*.json', recursive=True), leave = False): 

        xml_json_dict[json.load(open(filename, 'rb'))['paper_id']] = filename

    print(len(xml_json_dict), "papers from XML parsing")
def code_we_used_on_our_server() :

    tqdm.pandas()

    df["body_text"] = df.apply(lambda x: [], axis=1)

    df["sha"] = df["sha"] = df["sha"].apply(lambda x: x.split("; ")[0])

    for index, meta_data in tqdm(df.iterrows(), total = len(df.index)) :

        if meta_data["sha"] != "" and meta_data["sha"] in pdf_json_dict :

            file = json.load(open(pdf_json_dict[meta_data["sha"]], 'rb'))

            if(file['body_text'] != []) :

                df.at[(df[df['sha'] == meta_data["sha"]].index)[0], 'body_text'] = file['body_text']



        if meta_data["pmcid"] != "" and meta_data["pmcid"] in xml_json_dict :

            file = json.load(open(xml_json_dict[meta_data["pmcid"]], 'rb'))

            if(file['body_text'] != []) :

                df.at[(df[df['pmcid'] == meta_data["pmcid"]].index)[0], 'body_text'] = file['body_text']  
print("For the",len(df), "papers in the dataset (v7)")

print("-", len(df[df["abstract"] != ""]), "papers with abstract")
df['abstract'] = df['abstract'].apply(lambda x: re.sub('(\\n)+', ' ', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub('[a|A]bstract( [0-9]+)*', ' ', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub('ABSTRACT( [0-9]+)*', ' ', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub('[b|B]ackground(: )*', ' ', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub('BACKGROUND(: )*', ' ', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub('^[\s]*$', ' ', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub(r"http\S+", '', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub('"', '', x))

df['abstract'] = df['abstract'].apply(lambda x: re.sub("'", '', x))
df['abstract'].replace("", np.nan,inplace=True)

df.dropna(subset=['abstract'], inplace=True)

print("There are",len(df),"articles after removing missing values.")
df.drop_duplicates(subset=['abstract'], inplace=True)

print("There are",len(df),"articles after removing duplicate abstracts.")
df.drop(df.index[(df.abstract.str.len() < 100)],inplace=True)

print("There are",len(df),"articles after removing abstracts with few characters.")
display(df)
!pip install --user kleis-keyphrase-extraction
import kleis.resources.dataset as kl

from kleis.config.config import SEMEVAL2017
# load semeval 2017 dataset

dataset = kl.load_corpus(name=SEMEVAL2017)



# recomended options

dataset.training(features_method="simple-posseq", filter_min_count=16, tagging_notation="BILOU")
text_sample = df.sample()['abstract'].values[0]

keyphrases_sample = dataset.label_text(text_sample)

displacy.render({

        "text": text_sample,

        # keyphrases are in brat-like format (we use only the span)

        "ents": [{"start": start, "end": end, "label": "KP"}  for _, (_, (start, end)), _ in keyphrases_sample],

        "title": None

}, style="ent", options=options, manual=True)
special_chars = re.compile(r"\W")

underscores = re.compile(r"[_]+") 



def normalize_term(t):

    t_normalized, _ = re.subn(special_chars, "_", t.lower())

    t_normalized, _ = re.subn(underscores, "_", t_normalized)

    return t_normalized
def extract_abstract_kps(x) :

    return [(normalize_term(kptext), (start, end)) for _, (_, (start, end)), kptext in dataset.label_text(x)]
tqdm.pandas()

df["abstract_kps"] = df["abstract"].progress_apply(extract_abstract_kps)
display(df)
def get_kptext(list_kps):

    """Return keyphrase text"""

    return [ktext for kps in list_kps for ktext, _ in kps]



# initialize

df["tf"] = df["cord_uid"].apply(lambda x : {}) # it is later turned into a list



# keyphrase occurrences per doc

kps_docs_count = Counter() # |{d: k \in d}|

keyphrases = Counter()

for index, row in tqdm(df.iterrows(), total = len(df.index)):

    

    # list of keyphrases per document

    current_kps_count = Counter(get_kptext([row["abstract_kps"]]))

    

    # all keyphrases

    keyphrases.update(current_kps_count)

    

    # +1 for each keyphrase in the document

    kps_docs_count.update(current_kps_count.keys()) # |{d: k \in d}|

    

    # Keyphrase frequency # TF

    df.at[(df[df['cord_uid'] == row['cord_uid']].index)[0], 'tf'] = current_kps_count # Cnt
# keep cnt(keyphrase) > 1

keyphrases = {kp: cnt for kp, cnt in keyphrases.items() if cnt > 1}

# remove single symbols normalized

if "_" in keyphrases:

    del keyphrases["_"]



def get_tf(kps):

    total_count = sum(kps.values())

    return [(kp, cnt/total_count) for kp, cnt in kps.items() if kp in keyphrases]



tqdm.pandas()

df["tf"] = df["tf"].progress_apply(get_tf) # TF
N = len(df) # number of documents

tqdm.pandas()

df["idf"] = df["tf"].progress_apply(lambda x : [(kp, np.log(N/kps_docs_count[kp])) for kp, _ in x]) # IDF
def get_tf_idf(e):

    tf, idf = e

    return [(kp, ktf*idf[i][1]) for i, (kp, ktf) in enumerate(tf)]



tqdm.pandas()

df["tf-idf"] = df[['tf','idf']].progress_apply(lambda x: get_tf_idf(x), axis=1)
MAX_KEYPHRASES = 15



def rank_keyphrases(e):

    tf_idf, title, abstract = e

    # normalized title and abstract

    # underscores are added to search each _keyphrase_ as an single entity 

    # considering if it is the first or last word

    title = "_" + normalize_term(title) + "_" 

    abstract = "_" + normalize_term(abstract) + "_"

    # check if keyphrase is in title or abstract

    tf_idf = map(lambda e: (e[0], e[1] + (int(title.find("_" + e[0] + "_") >= 0) + int(abstract.find("_" + e[0] + "_") >= 0))/2 ), tf_idf)

    

    return sorted(tf_idf, key=lambda e: e[1], reverse=True)[:MAX_KEYPHRASES]

    

tqdm.pandas()

df["ranked_kps"] = df[["tf-idf", "title", "abstract"]].progress_apply(lambda x: rank_keyphrases(x), axis=1) # rank by tf-idf + title + abstract
tqdm.pandas()

df["keyphrases"] = df["ranked_kps"].progress_apply(lambda x: set(kp for kp, _ in x))
example = df.iloc[random.randrange(len(df))]

kps, tfidf = zip(*(example['ranked_kps']))

pandas.DataFrame({'ranked_kps': tfidf}, index=kps).iplot(kind='bar', title=example['title'])
nlp = spacy.load("en_core_web_sm", disable=["ner"])

nlp.pipe_names
MAX_SENTS = 5



df["ranked_sentences"] = df["cord_uid"].apply(lambda x : [])

df["doc_score_sents"] = df["cord_uid"].apply(lambda x : 0.0)

for index, row in tqdm(df.iterrows(), total = len(df.index)):

    

    # list of keyphrases per document

    paragraphs_texts = nlp.pipe(map(lambda par: par["text"], [{"text": row["abstract"]}]))

    

    # merge all keyphrases per paragraph

    full_kps = [row['abstract_kps']]

   

    # sum_sent_scores = 0.0

    sentences_count = 0

    scored_sentences = []

    # for each paragraph

    for i_par, par in enumerate(paragraphs_texts):

        

        # norm L1 of ranked keyphrases 

        l1_norm_rkps = sum(map(lambda x: x[1], row['ranked_kps'])) 

        

        # keyphrases in the curent paragraph

        paragraph_kps = full_kps[i_par]



        for s in par.sents:

            sentences_count += 1

            sent_start, sent_end = s.start_char, s.end_char

            s_text = s.text

            s_score = 0.0 # sentence score

            s_spans = []



            for rkp, kscore in row['ranked_kps']:

                # spans of current keyphrase

                kspans = [pkp_spans for pkp_text, pkp_spans in paragraph_kps if pkp_text == rkp]



                for kstart, kend in kspans:

                    if kstart >= sent_start and kstart <= sent_end:

                        s_score += (kscore/l1_norm_rkps)**2 # sum(X^2)

                        s_spans.append((kstart - sent_start, kend - sent_start))



            if s_score > 0.0:

                s_score = np.sqrt(s_score) # sqrt(sum(X^2)) # norm L2 normalized

                # sum_sent_scores += s_score



            scored_sentences.append((i_par, s_score, s_text, s_spans))

            

    # save ranked sentences

    ranked_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:MAX_SENTS]

    df.at[(df[df['cord_uid'] == row['cord_uid']].index)[0], 'ranked_sentences'] = ranked_sentences

    df.at[(df[df['cord_uid'] == row['cord_uid']].index)[0], 'doc_score_sents'] = sum([s_score for _, s_score, _, _ in ranked_sentences])/(sentences_count if sentences_count > 0 else 0.0)
# df["score"] = df.apply(lambda x: x['sum_sent_scores']/x['sentences_count'])

docs_avgsize = df['tf'].apply(lambda x: len(x)).sum()/len(df)



def okapi_bm25(d):

    """OKAPI BM25 over keyphrases"""

    # default values in elastic search

    okapi_b = 0.75

    okapi_k1 = 1.2



    kps, tf, idf = d

    tf = dict(tf)

    idf = dict(idf)

    

    doc_size = len(tf)

    okapi_denominator_part = okapi_k1 * (1 - okapi_b + okapi_b * doc_size/docs_avgsize)

    

    return sum([idf[kp] * (tf[kp] * (okapi_k1 + 1)) / (tf[kp] + okapi_denominator_part) for kp in kps])



tqdm.pandas()

df['bm25'] = df[['keyphrases', 'tf', 'idf']].progress_apply(okapi_bm25, axis=1)
del nlp

df.drop(columns=['abstract_kps'], inplace=True)
docs_avgsize = df['tf'].apply(lambda x: len(x)).sum()/len(df)



def okapi_bm25(d):

    """OKAPI BM25 over keyphrases"""

    # default values in elastic search

    okapi_b = 0.75

    okapi_k1 = 1.2



    kps, tf, idf = d

    kps = set(kp for kp, _ in kps)

    tf = dict(tf)

    idf = dict(idf)

    

    doc_size = len(tf)

    okapi_denominator_part = okapi_k1 * (1 - okapi_b + okapi_b * doc_size/docs_avgsize)

    

    return sum([idf[kp] * (tf[kp] * (okapi_k1 + 1)) / (tf[kp] + okapi_denominator_part) for kp in kps])



tqdm.pandas()

df['score'] = df[['ranked_kps', 'tf', 'idf']].progress_apply(okapi_bm25, axis=1)
df.drop(columns=['tf'], inplace=True)

df.drop(columns=['idf'], inplace=True)
phi = 1/3

bm25_max = df['score'].max()

score_sents_max = df['doc_score_sents'].max()

cnt_sents_max = MAX_SENTS



def get_doc_rank(e):

    bm25_score, score_sents, rsents = e

    return phi*(bm25_score/bm25_max + score_sents/score_sents_max + len(rsents)/cnt_sents_max)



tqdm.pandas()

df['score'] = df[['score', 'doc_score_sents', 'ranked_sentences']].apply(get_doc_rank, axis=1)
df.drop(columns=['doc_score_sents'], inplace=True)
df_nlargest = df.nlargest(10, 'score')



for index, row in tqdm(df_nlargest.iterrows(), total = len(df_nlargest.index)):

    print("\nDoc score: %0.4f" % row['score'])

    print("\n  Title: " + row['title'])

    for s in row['ranked_sentences']:

        i_par, s_score, text, spans = s

        print("\n[+] Sentence score: %0.4f" % s_score)

        displacy.render({

            "text": text,

            # keyphrases are in brat-like format (we use only the span)

            "ents": [{"start": start, "end": end, "label": "KP"}  for start,end in spans],

            "title": None

        }, style="ent", options=options, manual=True)
df.sort_values(by='score', ascending=False, inplace=True)
df.drop(df.loc[df['score']<=0.44].index, inplace = True)

df.reset_index(drop=True, inplace = True)
display(df)
def code_we_used_on_our_server() :

    model_path = "/path_to/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"

    model = sent2vec.Sent2vecModel()

    try:

        model.load_model(model_path)

    except Exception as e:

        print(e)

    print('model successfully loaded')

    

    stop_words = set(stopwords.words('english'))

    

    def preprocess_sentence(x):

        x = x.replace('/', ' / ')

        x = x.replace('.-', ' .- ')

        x = x.replace('.', ' . ')

        x = x.replace('\'', ' \' ')

        x = x.replace('[', ' [ ')

        x = x.replace(']', ' ] ')

        x = x.replace('(', ' ( ')

        x = x.replace(')', ' ) ')

        x = x.replace('%', ' % ')

        x = x.replace('"', ' " ')

        x = x.lower()



        return " ".join([token for token in word_tokenize(x) if token not in punctuation and token not in stop_words])

    

    tqdm.pandas()

    df_json['abstract_tokenised'] = df_json['abstract'].progress_apply(preprocess_sentence)

    df_json["abstract_vector"] = df_json["abstract_tokenised"].progress_apply(lambda x : model.embed_sentence(x))

    

    with open("/calcul/kaggle_challenge/CORD-19-research-challenge_v7/CORD-19_v7_final_data_0.36_vectors.tsv", 'w+') as tensors:

        with open("/calcul/kaggle_challenge/CORD-19-research-challenge_v7/CORD-19_v7_final_data_0.36_metadata.tsv", 'w+') as metadata:

            metadata.write("Index\tTitle\n")

            for index, row in tqdm(df_json.iterrows(), total = len(df_json.index)) :

                metadata.write("%d\t%s\n" % (index, row["title"].encode('utf8')))

                vector_row = '\t'.join(map(str, row["abstract_vector"][0]))

                tensors.write(vector_row + "\n")

    

    def most_similar_sentence(vec, num):

        sims = empty(len(df_json), dtype=float)

        vec_len = np_norm(vec)

        for idx, row in df_json.iterrows() :

            vec2 = row["abstract_vector"]

            vec2_len = np_norm(vec2)

            sims[idx] = np.dot(vec[0],vec2[0]) / vec_len / vec2_len

        nearest = []

        topN = argsort(sims)[::-1]

        display(topN)

        for top_sent in topN:

            if(idx != top_sent):

                nearest.append((top_sent,float(sims[top_sent])))

                if len(nearest) == num: break

        return nearest

    

    def apply_features(x):

        return most_similar_sentence(x, 10)



    df_json["KNN"] = df_json["abstract_vector"].parallel_apply(apply_features)