import pandas as pd
import sklearn 
import numpy as np
import nltk
#nltk.download('punkt')
import re
import time
import codecs
# import data using pandas and put into SFrames:
papers_data = pd.read_csv('../input/Papers.csv')
authors_data = pd.read_csv('../input/Authors.csv')
authorId_data = pd.read_csv('../input/PaperAuthors.csv')
def given_paperID_give_index(paper_id, paper_data):
    return paper_data[paper_data['Id']==paper_id].index[0]
#
def given_index_give_PaperID(index, paper_data):
    return paper_data.iloc[index]['Id']
Ex_paper_id = 5941
Ex_paper_index = given_paperID_give_index(Ex_paper_id, papers_data)
papers_data.iloc[Ex_paper_index]['PaperText'][0:1000]
def clean_text(text):
    list_of_cleaning_signs = ['\x0c', '\n']
    for sign in list_of_cleaning_signs:
        text = text.replace(sign, ' ')
    #text = unicode(text, errors='ignore')
    clean_text = re.sub('[^a-zA-Z]+', ' ', text)
    return clean_text.lower()
papers_data['PaperText_clean'] = papers_data['PaperText'].apply(lambda x: clean_text(x))
papers_data['Abstract_clean'] = papers_data['Abstract'].apply(lambda x: clean_text(x))
papers_data.iloc[Ex_paper_index]['PaperText_clean'][0:1000]
# here Brandon defines a tokenizer and stemmer which returns the set 
# of stems in the text that it is passed
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
from sklearn.feature_extraction.text import TfidfVectorizer
# Producing tf_idf matrix separately based on Abstract
tfidf_vectorizer_Abstract = TfidfVectorizer(max_df=0.95, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
%time tfidf_matrix_Abstract = tfidf_vectorizer_Abstract.fit_transform(papers_data['Abstract_clean'])

# Producing tf_idf matrix separately based on PaperText
tfidf_vectorizer_PaperText = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
%time tfidf_matrix_PaperText = tfidf_vectorizer_PaperText.fit_transform(papers_data['PaperText_clean'])
terms_Abstract = tfidf_vectorizer_Abstract.get_feature_names()
terms_PaperText = tfidf_vectorizer_Abstract.get_feature_names()
def top_tfidf_feats(row, terms, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(terms[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df['feature']
def given_paperID_give_keywords(paper_data, tfidfMatrix, terms, paper_id, top_n=20):
    row_id = given_paperID_give_index(paper_id, paper_data)
    row = np.squeeze(tfidfMatrix[row_id].toarray())
    return top_tfidf_feats(row, terms, top_n)
paper_id_example = 5941
print ("Keywords based on Abstract:")
print (given_paperID_give_keywords(papers_data, tfidf_matrix_Abstract,
                                  terms_Abstract, paper_id_example, top_n = 10))
from sklearn.neighbors import NearestNeighbors
# Based on Abstract
num_neighbors = 4
nbrs_Abstract = NearestNeighbors(n_neighbors=num_neighbors,
                                 algorithm='auto').fit(tfidf_matrix_Abstract)
distances_Abstract, indices_Abstract = nbrs_Abstract.kneighbors(tfidf_matrix_Abstract)
# Based on PaperText
nbrs_PaperText = NearestNeighbors(n_neighbors=num_neighbors,
                                  algorithm='auto').fit(tfidf_matrix_PaperText)
distances_PaperText, indices_PaperText = nbrs_PaperText.kneighbors(tfidf_matrix_PaperText)
print ("Nbrs of the example paper based on Abstract similarity: %r" % indices_Abstract[1])
print ("Nbrs of the example paper based on PaperText similarity: %r" % indices_PaperText[1])
Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("The Abstract of the example paper is:\n")
print (papers_data.iloc[indices_Abstract[Ex_index][0]]['Abstract'])
print ("The Abstract of the similar papers are:\n")
for i in range(1, len(indices_Abstract[Ex_index])):
    print ("Neighbor No. %r has following abstract: \n" % i)
    print (papers_data.iloc[indices_Abstract[Ex_index][i]]['Abstract'])
    print ("\n")
Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("The Abstract of the example paper is:\n")
print (papers_data.iloc[indices_PaperText[Ex_index][0]]['Abstract'])
print ("The Abstract of the similar papers are:\n")
for i in range(1, len(indices_PaperText[Ex_index])):
    print ("Neighbor No. %r has following abstract: \n" % i)
    print (papers_data.iloc[indices_PaperText[Ex_index][i]]['Abstract'])
    print ("\n")
def given_paperID_give_authours_id(paper_id, author_data, author_id_data):
    id_author_list = author_id_data[author_id_data['PaperId']==paper_id]['AuthorId']
    return id_author_list

def given_authorID_give_name(author_id, author_data):
    author_name = author_data[author_data['Id'] == author_id]['Name']
    return author_name

def given_similar_paperIDs_give_their_titles(sim_papers_list_index, paper_data):
    titles = []
    for index in sim_papers_list_index:
        titles.append(paper_data.iloc[index]['Title']+'.')
    return titles
Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("Title of similar papers to the example paper based on Abstract:\n\n")
for title in given_similar_paperIDs_give_their_titles(indices_Abstract[Ex_index], papers_data):
    print (title)
Ex_paper_id = 5941
Ex_index = given_paperID_give_index(Ex_paper_id, papers_data)
print ("Title of similar papers to the example paper based on Abstract:\n\n")
for title in given_similar_paperIDs_give_their_titles(indices_PaperText[Ex_index], papers_data):
    print (title)
