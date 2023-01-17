import os

import numpy as np

import pandas as pd 

from memory_profiler import profile

from typing import List

import pickle

from gensim.models.word2vec import Word2Vec



from tqdm.auto import tqdm

tqdm.pandas()

!pip install memory_utils
!pip install codeprofile
import memory_utils

from codeprofile import profiler
#limit number of documents if DOC_LIMIT != None. 

#allows testing the code without waiting 2 days, and to run a large set of questions in the notebook timelimit / resources

DOC_LIMIT = 42

DOC_SIZE_CAP = 100000 #caps document length at 100k characters for processing, saves memory and processing time
!pip install --upgrade transformers
from transformers import pipeline



nlp = pipeline("question-answering")
with open("/kaggle/input/covid-word2vec/word2vec.pickle", "rb") as f:

    w2v = pickle.load(f)
with open("/kaggle/input/covid-tfidf/i_index.pickle", "rb") as f:

    i_index = pickle.load(f)
with open("/kaggle/input/covid-tfidf/tfidf_matrix.pickle", "rb") as f:

    tfidf_matrix = pickle.load(f)
with open("/kaggle/input/covid-tfidf/doc_ids.pickle", "rb") as f:

    doc_ids = pickle.load(f)
#need to be able to load all the documents for the question-answering later, so load up all the paths first to identify where to find the document later.

import glob, os, json



def load_doc_paths():

    all_file_paths = []

    base_paths = [

        "/kaggle/input/CORD-19-research-challenge/arxiv/arxiv/pdf_json/*",

        "/kaggle/input/CORD-19-research-challenge/arxiv/arxiv/pmc_json/*",

        "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/*",

        "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pmc_json/*",

        "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/*",

        "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/*",

        "/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/*",

        "/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json/*",

        "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/*",

        "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/*",

    ]

    for base_path in base_paths:

        file_paths_glob = glob.glob(base_path)

        all_file_paths.extend(file_paths_glob)

    return all_file_paths
all_doc_paths = load_doc_paths()
df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

df_metadata.head()
i_index["patient"]
def build_dicts(threshold):

    word_weight_dicts = {}

    for word in tqdm(i_index.keys()):

        doc_weights = i_index[word]

        doc_weight_dict = {}

        word_weight_dicts[word] = doc_weight_dict

        for doc_idx, doc_weight in doc_weights:

            doc_idx = int(doc_idx)

            doc_weight_dict[doc_idx] = doc_weight

            #reduct sizes by capping on some number of docs

            if doc_weight < threshold and len(doc_weight_dict) > 1000:

                break

    return word_weight_dicts

            
w2v.init_sims()
w2v.similar_by_vector("patient", 10)
def find_pairs(words_sentence):

    words = words_sentence.split(" ")

    word_lists = []

    for word in words:

        synonyms = w2v.similar_by_vector(word, topn = 10)

        selected = [word]

        for synonym in synonyms:

            #if word2vec distance factor is less than 0.5, stop adding. expect input to be sorted..

            if synonym[1] < 0.5:

                break

            selected.append(synonym[0])

        word_lists.append(selected)

    

    for word_list in word_lists:

        print("synonyms found:")

        print(f"{word_list[0]}: {word_list[1:]}")

    

    return word_lists

def find_weight_threshold():

    arrays = [weights for weights in i_index.values()]

    print(f"loaded doc weights for {len(arrays)} words.")

    all_data = np.concatenate(arrays)

    print(f"a total of {all_data.shape} doc-word weights loaded.")

    d_mean = np.mean(all_data[:,1])

    d_med = np.median(all_data[:,1])

    d_max = np.max(all_data[:,1])

    d_min = np.min(all_data[:,1])

    p80 = np.percentile(all_data[:,1],80)

    p70 = np.percentile(all_data[:,1],70)

    p30 = np.percentile(all_data[:,1],30)

    print(f"min={d_min}, max={d_max}\n"+

          f"avg={d_mean}, median={d_med}\n"+

          f"p30={p30}, p80=(p80)")

    with np.printoptions(precision=20, suppress=True):

        print(np.array([d_min, d_max, d_mean, d_med, p30, p80]))

    #threshold = np.max([d_mean, d_med])

    threshold = p70 #using 70 to limit the size of the notebook

    print(f"threshold: {threshold}")

    return threshold
%time

threshold = find_weight_threshold()

word_dicts = build_dicts(threshold)

from collections import defaultdict



@profiler.profile_func

def find_docs_for_words(synonym_lists):

    total_scores = defaultdict(lambda: [])

    #assume query (keyword) string "incubation period"

    #word_list1 = incubation and its synonyms

    #word_list2 = period and its synonyms

    for word_list in synonym_lists:

        doc_scores = defaultdict(lambda: [])

        #each word list represent one base word and its synonyms, 

        #so first sum up all weights for a single word and its synonyms

        #note that above we filtered by threshold and number of words, so should not have very small scores in it

        #TODO: improve score weights and filter count threshold

        for word in word_list:

            if word not in word_dicts:

                #some of the synonyms from word2vec may be rare and were dropped by earlier preprocessing steps

                #this prints those so we can see if it is a real loss (should we go back and add it) or not

                print(f"missed word: {word}")

                continue

            word_doc_scores = word_dicts[word]

            for doc_idx in word_doc_scores:

                #get weights for this word for each document the word appears in

                doc_scores[doc_idx].append(word_doc_scores[doc_idx])

        for doc_idx in doc_scores:

            #sum up all synonyms into one score for each document. 

            #after this each doc has as many lists as it has base words with weights

            total_scores[doc_idx].append(sum(doc_scores[doc_idx]))

        #so at this point total_scores has one entry per doc_id: (weights1, weights2). 

        #if there are less than 2, it did not have weight in one of the two, and will be removed later

        #TODO: nicer filtering schema        

    return total_scores
@profiler.profile_func

def filter_weighted_docs(n_words, total_scores):

    #remove all docs that do not have a weight for one of the N base words

    to_remove = []

    for doc_id in total_scores:

        ds = total_scores[doc_id]

        if len(ds) < n_words:

            to_remove.append(doc_id)

            continue

        total_scores[doc_id] = sum(ds)

    print(f"removing {len(to_remove)} docs")

    for key in to_remove:

        del total_scores[key]

@profiler.profile_func

def doc_scores_to_df(total_scores):

    #NOTE: here we sort the docs by score so from this on highest scoring will be first

    #filter_weighted_docs has summed all the word weights for keywords into one, stores in item[1] here 

    ts_dict = {k: v for k, v in sorted(total_scores.items(), key=lambda item: item[1], reverse=True)}

    #print(ts_dict)

    df_ts = pd.DataFrame(ts_dict.items(), columns=['DocID', 'WeightScore'])

    return df_ts
@profiler.profile_func

def get_doc_ids_to_load(doc_ids, df_total_scores):

    doc_ids_to_load = []

    for index, row in df_total_scores.iterrows():

        doc_idx = int(row["DocID"])

        #print(doc_idx)

        doc_ids_to_load.append(doc_ids[doc_idx])

    return doc_ids_to_load
@profiler.profile_func

def load_docs(doc_ids_to_load, filepaths, df_metadata):

    loaded_docs = {}

    new_ids = []

    

    for doc_id in tqdm(doc_ids_to_load):

        if doc_id in loaded_docs:

            print(f"WARNING: duplicate doc id to load: {doc_id}, skipping")

            continue

         

        #TODO: this should not work if SHA is nan, why does it?

        doc_sha = df_metadata[df_metadata["cord_uid"] == doc_id]["sha"]

        if doc_sha.shape[0] > 0:

            doc_sha = doc_sha.values[0]

        else:

            doc_sha = None

        #print(doc_sha)

        #TODO: this should not work if PMCID is nan, why does it?

        doc_pmcid = df_metadata[df_metadata["cord_uid"] == doc_id]["pmcid"]

        if doc_pmcid.shape[0] > 0:

            doc_pmcid = doc_pmcid.values[0]

        else:

            doc_pmcid = None

        pmc_path = None

        sha_path = None

        for filepath in filepaths:

            if isinstance(doc_pmcid, str) and doc_pmcid in filepath:

                pmc_path = filepath

                break

            if isinstance(doc_sha, str) and doc_sha in filepath:

                sha_path = filepath

        if pmc_path is not None:

            #always favour PMC docs since they are described as higher quality (not scanned from PDF but direct machine format)

            filepath = pmc_path

        else:

            filepath = sha_path

        #print(filepath)

        if filepath is None:

            print(f"WARNING: cannot find path for doc id {doc_id}. Possibly Kaggle dataset has changed?")

            continue

        with open(filepath) as f:

            d = json.load(f)

            body = ""

            for idx, paragraph in enumerate(d["body_text"]):

                body += f"{paragraph['text']}\n"

                #print(paragraph)

                #print("---------")

            loaded_docs[doc_id] = body

            new_ids.append(doc_id)

            

    return loaded_docs, new_ids
@profiler.profile_func

def find_docs_for_query(query):

    print(f"query: {query}")

    pairs = find_pairs(query)

    n_words = len(pairs)

    print(f"query has {n_words} words")

    total_scores = find_docs_for_words(pairs)

    #print(total_scores)

    print(f"number of docs with some search terms (at high score): {len(total_scores)}")

    filter_weighted_docs(n_words, total_scores)

    print(f"number of docs with all search terms (at high score): {len(total_scores)}")

    #this also sorts the scores before creating the dataframe. so highest scoring are first

    df_scores = doc_scores_to_df(total_scores)

    query_doc_ids = get_doc_ids_to_load(doc_ids, df_scores)

    print(f"num. doc ids to load for the final docs: {len(query_doc_ids)}")

    if DOC_LIMIT is not None:

        #this avoid the overhead of loading thousands of extra documents. memory+processing time

        #the multiplier is to give it an extra chance to find answers that meet the confidence level

        query_doc_ids = query_doc_ids[:DOC_LIMIT*2]

    print(f"num. doc ids to load for the final docs after capping: {len(query_doc_ids)}")

    loaded_docs, query_doc_ids = load_docs(query_doc_ids, all_doc_paths, df_metadata)

    return query_doc_ids, loaded_docs



@profiler.profile_func

def run_query(loaded_docs, doc_ids, question):

    scores = []

    answers = []

    processed_ids = set()

    for doc_id in tqdm(doc_ids):

        if doc_id in processed_ids:

            print(f"skipping already processed doc id: {doc_id}")

            continue

        processed_ids.add(doc_id)

        #doc_id = doc_ids[idx]

        context = loaded_docs[doc_id]

        #print(len(scores))

        #memory_utils.print_memory()

        #print(len(context))

        context = context[:DOC_SIZE_CAP]

        #print(len(context))

        if context is None:

            print(f"skipping doc id {doc_id}, not found")

            continue

        with profiler.profile("nlp question"):

            answer = nlp(question=question, context=context)

        score = answer["score"]

        answer_text = answer["answer"]

        print(f"question: {question}")

        print(f"  doc id: {doc_id}")

        print(f"  answer: {answer_text}")

        print(f"  score: {score}")

        scores.append(score)

        answers.append(answer_text)

        if DOC_LIMIT is not None and len(answers) > DOC_LIMIT:

            break

    return scores, answers
@profiler.profile_func

def build_results_df(query_doc_ids, df_metadata, scores, answers, score_limit):

    titles = []

    publish_times = []

    journals = []

    author_lists = []

    filtered_scores = []

    filtered_answers = []

    for idx, doc_id in enumerate(query_doc_ids):

        if DOC_LIMIT is not None and idx > DOC_LIMIT:

            break

        

        doc_meta = df_metadata[df_metadata["cord_uid"] == doc_id]

        title = doc_meta["title"].values[0]

        publish_time = doc_meta["publish_time"].values[0]

        journal = doc_meta["journal"].values[0]

        authors = doc_meta["authors"].values[0]

        score = scores[idx]

        answer = answers[idx]

        if (score < score_limit):

            continue



        titles.append(title)

        publish_times.append(publish_time)

        journals.append(journal)

        author_lists.append(authors)

        filtered_scores.append(score)

        filtered_answers.append(answer)

    

    df_result = pd.DataFrame({

        "Article title": titles,

        "Published": publish_times,

        "Journal": journals,

        "Authors": author_lists,

        "Confidence": filtered_scores,

        "Answer": filtered_answers

    })

    return df_result

    
def answer_a_question(tfidf_sentence, question, df_metadata, score_limit):

    query_doc_ids, query_docs = find_docs_for_query(tfidf_sentence)

    print(f"doc_ids={len(query_doc_ids)}, docs={len(query_docs)}")

    scores, answers = run_query(query_docs, query_doc_ids, question)

    df = build_results_df(query_doc_ids, df_metadata, scores, answers, score_limit)

    return df
pd.set_option('display.max_rows', 1000)
memory_utils.print_memory()
q = "What is the incubation period?"

df = answer_a_question("covid19 incubation period",  q, df_metadata, 0.7)
print(f"Q: {q}")

df
q = "How prevalent is asymptomatic shedding and transmission?"

df = answer_a_question("asymptomatic shedding transmission",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "What is the transmission seasonality?"

df = answer_a_question("covid19 transmission seasonality",  q, df_metadata, 0.02)
print(f"Q: {q}")

df
q = "What is covid19 chemical structure?"

df = answer_a_question("covid19 chemical structure",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "How long does covid19 survive?"

df = answer_a_question("covid19 persistent host",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "How long is host infectous?"

df = answer_a_question("covid19 infect host",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "What does covid19 persist on?"

df = answer_a_question("covid19 copper steel plastic",  q, df_metadata, 0.005)
print(f"Q: {q}")

df
q = "What is the history of covid19?"

df = answer_a_question("covid19 history",  q, df_metadata, 0.05)
print(f"Q: {q}")

df
q = "What is the disease model?"

df = answer_a_question("covid19 disease model",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "What are effective diagnostics processes?"

df = answer_a_question("covid19 diagnostic process",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "How does the virus change and adapt?"

df = answer_a_question("covid19 phenotypic change adaptation",  q, df_metadata, 0.01)
print(f"Q: {q}")

df
q = "What is effective movement control strategy?"

df = answer_a_question("covid19 movement control strategy",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "What is effective protective equipment?"

df = answer_a_question("covid19 personal protective equipment",  q, df_metadata, 0.01)
print(f"Q: {q}")

df
q = "What is the role of environment in transmission?"

df = answer_a_question("covid19 environment transmission",  q, df_metadata, 0.05)
print(f"Q: {q}")

df
q = "What is immune response?"

df = answer_a_question("covid19 immune response",  q, df_metadata, 0.05)
print(f"Q: {q}")

df
q = "How long is immunity?"

df = answer_a_question("covid19 immunity period",  q, df_metadata, 0.05)
print(f"Q: {q}")

df
q = "What are risk factors?"

df = answer_a_question("covid19 risk factor",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "Does smoking increase risk?"

df = answer_a_question("covid19 smoke risk",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "How does coninfection affect transmission?"

df = answer_a_question("covid19 coinfection transmission",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "What is risk to pregnant women?"

df = answer_a_question("covid19 pregnant woman",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "Does social status affect risk?"

df = answer_a_question("covid19 social economic",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "How is the virus transmitted?"

df = answer_a_question("covid19 transmission dynamic",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "What is the reproductive number?"

df = answer_a_question("covid19 reproduction number",  q, df_metadata, 0.7)
print(f"Q: {q}")

df
q = "How is covid19 transmitted?"

df = answer_a_question("covid19 transmission mode",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "What the impact of environment on transmission?"

df = answer_a_question("covid19 environment factor",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "What is the severity of covid19?"

df = answer_a_question("covid19 severity risk",  q, df_metadata, 0.03)
print(f"Q: {q}")

df
q = "Who is most at risk?"

df = answer_a_question("covid19 risk population",  q, df_metadata, 0.01)
print(f"Q: {q}")

df
q = "What are effective mitigation measures?"

df = answer_a_question("covid19 mitigation measure",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "What the asymptotic fatality rate?"

df = answer_a_question("covid19 asymptotic fatality",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "What drugs are effective?"

df = answer_a_question("covid19 drug effective",  q, df_metadata, 0.4)
print(f"Q: {q}")

df
q = "How effective are drugs?"

df = answer_a_question("covid19 drug effective", q, df_metadata, 0.4)
print(f"Q: {q}")

df
q = "What inhibitors are effective?"

df = answer_a_question("covid19 viral inhibitor",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "What is the effectiveness of inhibitors?"

df = answer_a_question("covid19 viral inhibitor",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "How effective is antibiotic enhancement?"

df = answer_a_question("covid19 antibiotic enhancement",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "Are animal models effective for humans?"

df = answer_a_question("covid19 animal model",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
#q = "How can production capacity be expanded?"

#df = answer_a_question("covid19 production capacity",  q, df_metadata, 0.1)
#print(f"Q: {q}")

#df
q  = "How to get people to use masks?"

df = answer_a_question("covid19 mask respirator",  q, df_metadata, 0.15)
print(f"Q: {q}")

df
q  = "What is effective vaccine?"

df = answer_a_question("covid19 vaccine develop",  q, df_metadata, 0.35)
print(f"Q: {q}")

df
q = "What are vaccine risks?"

df = answer_a_question("covid19 vaccine risk",  q, df_metadata, 0.01)
print(f"Q: {q}")

df
q = "How can we support nursing facilities?"

df = answer_a_question("covid19 support nurse facility",  q, df_metadata, 0.4)
print(f"Q: {q}")

df
q = "Does age have effect?"

df = answer_a_question("organ failure mortality",  q, df_metadata, 0.2)
print(f"Q: {q}")

df
q = "How do ethical principles map to covid19?"

df = answer_a_question("covid19 ethical principle",  q, df_metadata, 0.4)
print(f"Q: {q}")

df
q = "What do people fear?"

df = answer_a_question("covid19 fear anxiety",  q, df_metadata, 0.4)
print(f"Q: {q}")

df
q = "What are local barriers and enablers?"

df = answer_a_question("covid19 barrier enabler",  q, df_metadata, 0.4)
print(f"Q: {q}")

df
#q = "How are healthcare providers affected?"

#df = answer_a_question("covid19 provider health psychological",  q, df_metadata, 0.4)
#print(f"Q: {q}")

#df
q = "what is the experiment outcome?"

df = answer_a_question("membrane oxygenation",  q, df_metadata, 0.02)
print(f"Q: {q}")

df
q = "Is heart attack more likely?"

df = answer_a_question("covid19 heart attack",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
q = "How does regulation affect care level?"

df = answer_a_question("covid19 regulatory regulation",  q, df_metadata, 0.1)
print(f"Q: {q}")

df
#q = "How to get people to use masks?"

#df = answer_a_question("covid19 mask respirator",  q, df_metadata, 0.15)
print(f"Q: {q}")

df
q = "How to provide effective remote support?"

df = answer_a_question("covid19 telemedicine support",  q, df_metadata, 0.35)
print(f"Q: {q}")

df
q = "What can people do at home?"

df = answer_a_question("covid19 home guidance",  q, df_metadata, 0.01)
print(f"Q: {q}")

df
q = "What are effective diagnostics processes?"

df = answer_a_question("covid19 diagnostic process",  q, df_metadata, 0.002)
print(f"Q: {q}")

df
q = "What oral medication works?"

df = answer_a_question("covid19 oral medication",  q, df_metadata, 0.35)
print(f"Q: {q}")

df
q = "What are best practices at hospitals?"

df = answer_a_question("covid19 hospital best practice",  q, df_metadata, 0.15)
print(f"Q: {q}")

df
q = "What is effective protective equipment?"

df = answer_a_question("covid19 personal protective equipment",  q, df_metadata, 0.4)
print(f"Q: {q}")

df
#q = "How is artificial intelligence used?"

#df = answer_a_question("covid19 intervention automation",  q, df_metadata, 0.1)
#print(f"Q: {q}")

#df
profiler.print_run_stats()