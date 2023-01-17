import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from nltk.tokenize import sent_tokenize
import scipy.spatial

!pip install -U sentence-transformers

# Library from here: https://github.com/UKPLab/sentence-transformers
from sentence_transformers import SentenceTransformer
CLEAN_DATA_PATH = "../input/cord-19-eda-parse-json-and-generate-clean-csv/"

biorxiv_df = pd.read_csv(CLEAN_DATA_PATH + "biorxiv_clean.csv")
clean_pmc = pd.read_csv(CLEAN_DATA_PATH + "clean_pmc.csv")
papers_df = pd.concat([clean_pmc, biorxiv_df], axis=0).reset_index(drop=True)

papers_df.dropna(inplace=True)
papers_df.drop_duplicates(subset=['title'], keep=False, inplace=True)

# Load Sentence Embedding Model.
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
papers_df
sentences = sent_tokenize(papers_df.iloc[0]['text'])
sentences_df = pd.DataFrame({'id':np.zeros(len(sentences)).astype(int), 'sentences':sentences},index=None)

for i in range(1, len(papers_df)):
    paper_sentences = sent_tokenize(papers_df.iloc[i]['text'])
    paper_sentences_df = pd.DataFrame({'id':(np.ones(len(paper_sentences))*i).astype(int), 'sentences':paper_sentences},index=None)
    sentences_df = pd.concat([sentences_df, paper_sentences_df], axis=0).reset_index(drop=True)
    
sentences = sentences_df['sentences'].str.lower().tolist()
sentence_embeddings = model.encode(sentences)
question_1 = 'medical health care covid-19 coronavirus'
question_1_embedding = model.encode(question_1)

question_sentence_similarity_scores = []
for i in range(len(sentence_embeddings)):
    question_sentence_similarity_scores.append(scipy.spatial.distance.cdist([question_1_embedding[0]], [sentence_embeddings[i]], "cosine")[0])
sentences_df['cosine_score'] = question_sentence_similarity_scores

sentences_df.head()
for index, row in sentences_df[sentences_df['cosine_score'] > 1.25].iterrows():
    print(f"ID: {index}\nSentence: {row['sentences']}", '\n')
