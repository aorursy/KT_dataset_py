import collections
import csv
import datetime
import json
import os
import pickle
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
from matplotlib import pyplot
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
def extract_data_from_single_json(file_name: str) -> Tuple[str, str, str]:
    with open(file_name, 'r', encoding='utf-8') as file_handle:
        parsed_json = json.load(file_handle)

    paper_id, title, abstract, body = '', '', '', ''

    if 'paper_id' in parsed_json:
        paper_id = parsed_json['paper_id']

    if 'metadata' in parsed_json and 'title' in parsed_json['metadata']:
        title = parsed_json['metadata']['title']

    if 'abstract' in parsed_json:
        abstract = " ".join([abstract['text'] for abstract in parsed_json['abstract']])

    if 'body_text' in parsed_json:
        body = " ".join([body['text'] for body in parsed_json['body_text']])

    # some papers are parsed directly from Latex and have a lot of noise -> filter those
    if '\\usepackage' in abstract or '\\usepackage' in body:
        paper_id, title, abstract, body = '', '', '', ''

    # remove really short papers
    if len(abstract) + len(body) < 100:
        paper_id, title, abstract, body = '', '', '', ''

    return paper_id, title, abstract + " " + body
def get_data_in_one_folder(path: str) -> Tuple[List[str], List[str], List[str]]:
    file_names = os.listdir(path)
    file_names = map(lambda file_name: os.path.join(path, file_name), file_names)
    data = map(extract_data_from_single_json, file_names)
    # check if the paper_id is empty <=> something happened during reading
    data = filter(lambda t: t[0].strip() != '', data)
    return tuple(zip(*data))
def get_all_data() -> Tuple[List[str], List[str], List[str]]:
    all_paths = [
        '../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',
        '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',
        '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json',
        '../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json',
        '../input/CORD-19-research-challenge/custom_license/custom_license/pmc_json',
        '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json',
        '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json'
    ]
    all_paper_ids, all_titles, all_texts = [], [], []
    for path in all_paths:
        ids, titles, texts = get_data_in_one_folder(path)
        all_paper_ids += ids
        all_titles += titles
        all_texts += texts
        print("{} has been processed".format(path))
    return all_paper_ids, all_titles, all_texts
def train_model() -> Tuple[List[str], sparse.csr.csr_matrix, List[str], TfidfVectorizer]:
    paper_ids, titles, corpus_text = get_all_data()
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        token_pattern='\\b\\w*[a-zA-Z]\\w*\\b',  # numbers don't constitute a valid token
        max_df=0.5, min_df=40
    )
    tfidf_vectorizer.fit(corpus_text)
    tfidf_matrix = tfidf_vectorizer.transform(corpus_text)
    
    return paper_ids, tfidf_matrix, titles, tfidf_vectorizer
def get_indices_with_highest_similarity(query, tfidf_matrix, k: int) -> List[Tuple[float, int]]:
    cosine_similarities = cosine_similarity(tfidf_matrix, query).reshape(-1)
    top_indices = np.argpartition(cosine_similarities, -k - 1)[-k - 1:]
    indices_with_highest_similarity = [(cosine_similarities[i], i) for i in top_indices]
    indices_with_highest_similarity.sort(reverse=True)
    return indices_with_highest_similarity
# this is the final piece of code to create the model
paper_ids, tfidf_matrix, titles, tfidf_vectorizer = train_model()
print("the training has finished")
vocabulary = tfidf_vectorizer.get_feature_names()
print("{} words are used".format(len(vocabulary)))
print("{} documents are used".format(len(titles)))
k_most_similar = 5

# get the query
paper_from_corpus_is_query = False
query_is_ok = True
query = input("\033[1;37;40m Enter your query-text (can also be the ID of a paper in the corpus): ")
if query in paper_ids:
    index = paper_ids.index(query)
    feature_vector = tfidf_matrix[index].toarray()
    paper_from_corpus_is_query = True
else:
    feature_vector = tfidf_vectorizer.transform([query])
    if feature_vector.count_nonzero() > 0:
        feature_vector = feature_vector.toarray()
    else:
        print("The entered text couldn't be resolved to be a valid ID nor contains tokens from our vocabulary")
        query_is_ok = False

if query_is_ok:
    # note: we look for the '3 * k_most_similar' papers to then try and filter out duplicates (but this is not the most elegant way to do this)
    best_papers_scores_and_indices: List[Tuple[float, int]] = \
        get_indices_with_highest_similarity(feature_vector, tfidf_matrix, k=3 * k_most_similar)

    # print the similar papers, trying to remove duplicates
    print("\nThe {} most similar papers are:".format(k_most_similar))
    i = 0
    # if we use a paper from the corpus as query, we want to remove any mentions of itself in the top similar papers
    old_titles = [titles[index] if paper_from_corpus_is_query and titles[index].strip() != '' else None]
    for score, c_index in best_papers_scores_and_indices:
        if titles[c_index] in old_titles:
            continue
        old_titles.append(titles[c_index] if titles[c_index].strip() != '' else None)
        print("{}. id: {}\t(title: {}\tsimilarity score: {})".format(
            i+1, paper_ids[c_index], titles[c_index] if titles[c_index].strip() != '' else '*no title available*', score
        ))
        i += 1
        if i >= k_most_similar:
            break
# note: the full calculation takes a very long time. Therefore a subset of the corpus is sampled.
# if you have the compute or patience to compute the t-SNE fit for the whole dataset, you can change the code

tfidf_text_emb = tfidf_vectorizer.transform(["risk factor smoking"])

np.random.seed(1)
sampled_indices = np.random.choice(tfidf_matrix.shape[0], 20_000)
tsne = TSNE(n_components=2, n_iter=1000, random_state=1)
emb_visualised = tsne.fit_transform(sparse.vstack([tfidf_matrix[sampled_indices], tfidf_text_emb]))

pyplot.scatter(
    emb_visualised[:, 0], emb_visualised[:, 1],
    s=[0.64]*(emb_visualised.shape[0] - 1) + [25], color=['blue']*(emb_visualised.shape[0] - 1) + ['red']
)
pyplot.show()
def adjust_word_importance(feature_vector, vocabulary: List[str], importance_multiplier: Union[float, int]) -> np.ndarray:
    important_words = input(
        "\033[1;37;40m Which words do you want to emphasize (separate with whitespaces)? (will multiply the weight by {}) ".format(importance_multiplier)
    ).split(',')
    important_words = filter(None, map(lambda w: w.strip(), important_words))
    important_words_to_count = collections.Counter(important_words)
    for important_word, count in important_words_to_count.items():
        if important_word in vocabulary:
            feature_vector[0, vocabulary.index(important_word)] = \
                (importance_multiplier ** count) * feature_vector[0, vocabulary.index(important_word)]
        else:
            print('{} is not in our vocabulary'.format(important_word))
    feature_vector = feature_vector / np.linalg.norm(feature_vector)
    return feature_vector
k_most_similar = 5
importance_multiplier = 2

# get the query
paper_from_corpus_is_query = False
query_is_ok = True
query = input("\033[1;37;40m \n\n\nEnter your query-text (can also be the ID of a paper in the corpus): ")
if query in paper_ids:
    index = paper_ids.index(query)
    feature_vector = tfidf_matrix[index].toarray()
    paper_from_corpus_is_query = True
else:
    feature_vector = tfidf_vectorizer.transform([query])
    if feature_vector.count_nonzero() > 0:
        feature_vector = feature_vector.toarray()
    else:
        print("The entered text couldn't be resolved to be a valid "
              "ID nor contains it tokens from our vocabulary")
        query_is_ok = False

if query_is_ok:
    # the user may want to tweak the features a little
    feature_vector = adjust_word_importance(feature_vector, vocabulary, importance_multiplier)

    # note: we look for the '3 * k_most_similar' papers to then try and filter out duplicates (but this is not the most elegant way to do this)
    best_papers_scores_and_indices: List[Tuple[float, int]] = \
        get_indices_with_highest_similarity(feature_vector, tfidf_matrix, k=3 * k_most_similar)

    # print the similar papers, trying to remove duplicates
    print("\nThe most similar papers are:")
    i = 0
    # if we use a paper from the corpus as query, we want to remove any mentions of itself in the top similar papers
    old_titles = [titles[index] if paper_from_corpus_is_query and titles[index].strip() != '' else None]
    for score, c_index in best_papers_scores_and_indices:
        if titles[c_index] in old_titles:
            continue
        old_titles.append(titles[c_index] if titles[c_index].strip() != '' else None)
        print("id: {}\t(title: {}\tsimilarity score: {})".format(
            paper_ids[c_index], titles[c_index] if titles[c_index].strip() != '' else '*no title available*', score
        ))
        i += 1
        if i >= k_most_similar:
            break
def knock_out_dominant_feature(dominant_feature_threshold, feature_vector, vocabulary) -> Tuple[bool, np.ndarray]:
    index_2nd_largest_feature, index_largest_feature = np.argpartition(feature_vector.reshape(-1), -2)[-2:]
    if feature_vector[0, index_largest_feature] - feature_vector[0, index_2nd_largest_feature] \
            >= dominant_feature_threshold and feature_vector[0, index_2nd_largest_feature] >= 0.005:
        user_can_knock_out = True
        print(
            "\nThe word '{}' has the largest tfidf value ({}) in your paper and is much higher than the second "
            "highest ('{}' with {})".format(
                vocabulary[index_largest_feature], feature_vector[0, index_largest_feature],
                vocabulary[index_2nd_largest_feature], feature_vector[0, index_2nd_largest_feature]
            )
        )

        verdict = input(
            "\033[1;37;40m Do you want to leave this word out for now (enter r), divide its weight by a factor (enter the divider "
            "(needs to be an int)), or proceed (enter nothing)? "
        )

        user_wants_to_knock_out = verdict == 'r' or (verdict.isnumeric() and int(verdict) != 0)

        if verdict == 'r':
            feature_vector[0, index_largest_feature] = 0
        elif verdict.isnumeric() and int(verdict) != 0:
            feature_vector[0, index_largest_feature] = feature_vector[0, index_largest_feature] / int(verdict)
    else:
        user_can_knock_out = False
    return (user_can_knock_out and user_wants_to_knock_out), feature_vector / np.linalg.norm(feature_vector)
k_most_similar = 5
dominant_feature_threshold = 0.4
importance_multiplier = 2

# get the query
paper_from_corpus_is_query = False
query_is_ok = True
query = input("\033[1;37;40m \n\n\nEnter your query-text (can also be the ID of a paper in the corpus): ")
if query in paper_ids:
    index = paper_ids.index(query)
    feature_vector = tfidf_matrix[index].toarray()
    paper_from_corpus_is_query = True
else:
    feature_vector = tfidf_vectorizer.transform([query])
    if feature_vector.count_nonzero() > 0:
        feature_vector = feature_vector.toarray()
    else:
        print("The entered text couldn't be resolved to be a valid "
              "ID nor contains it tokens from our vocabulary")
        query_is_ok = False

if query_is_ok:
    # the user may want to tweak the features a little
    feature_vector = adjust_word_importance(feature_vector, vocabulary, importance_multiplier)

    knock_out_happening = True
    while knock_out_happening:
        # note: we look for the '3 * k_most_similar' papers to then try and filter out duplicates 
        # (but this is not the most elegant way to do this)
        best_papers_scores_and_indices: List[Tuple[float, int]] = \
            get_indices_with_highest_similarity(feature_vector, tfidf_matrix, k=3 * k_most_similar)

        # print the similar papers, trying to remove duplicates
        print("\nThe most similar papers are:")
        i = 0
        # if we use a paper from the corpus as query, we want to remove any mentions of itself in the top similar papers
        old_titles = [titles[index] if paper_from_corpus_is_query and titles[index].strip() != '' else None]
        for score, c_index in best_papers_scores_and_indices:
            if titles[c_index] in old_titles:
                continue
            old_titles.append(titles[c_index] if titles[c_index].strip() != '' else None)
            print("id: {}\t(title: {}\tsimilarity score: {})".format(
                paper_ids[c_index], titles[c_index] if titles[c_index].strip() != '' else '*no title available*', score
            ))
            i += 1
            if i >= k_most_similar:
                break

        knock_out_happening, feature_vector = \
            knock_out_dominant_feature(dominant_feature_threshold, feature_vector, vocabulary)