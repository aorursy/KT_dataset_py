import numpy as np 
import pandas as pd
import glob
import json
import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from spacy.lang.en.stop_words import STOP_WORDS

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from collections import Counter
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            self.title = content['metadata']['title']
            self.authors = ''
            for author in content['metadata']['authors']:
                #concatenate first name and last name
                self.authors += author['first'] + ' ' + author['last'] + ", "
            #removing last comma and a space from string authors
            self.authors = self.authors[:-2]
            # Abstract
            try:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except:
                self.abstract = []
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            #converting list to string
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)
dict_ = {'paper_id': [],  'title': [], 'authors': [], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['title'].append(content.title)
    dict_['authors'].append(content.authors)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'title', 'authors', 'abstract', 'body_text'])
df_covid.head()
temp = df_covid[df_covid['title']=='Forty years with coronaviruses'].copy()
temp.drop_duplicates(['abstract', 'body_text'], inplace=True)
temp
temp.iloc[0]['body_text'] == temp.iloc[1]['body_text']
df_covid.dropna(inplace=True)
df_covid.info()
df_covid_small = df_covid.head(10000).copy()
df_covid_small['filtered_body_text'] = df_covid_small['body_text'].copy()
#remove square bracket contents
df_covid_small['filtered_body_text'] = df_covid_small['filtered_body_text'].apply(lambda x: re.sub('\[.*?\]','',x))
#remove parenthesis contents
df_covid_small['filtered_body_text'] = df_covid_small['filtered_body_text'].apply(lambda x: re.sub('\(.*?\)','',x))
#remove punctuation except hyphen
df_covid_small['filtered_body_text'] = df_covid_small['filtered_body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s-]','',x))
#remove \n \t
df_covid_small['filtered_body_text'] = df_covid_small['filtered_body_text'].apply(lambda x: re.sub('\n|\t',' ',x))
#remove extra space
df_covid_small['filtered_body_text'] = df_covid_small['filtered_body_text'].apply(lambda x: re.sub(' +',' ',x))
# make everything lower case
df_covid_small['filtered_body_text'] = df_covid_small['filtered_body_text'].apply(lambda x: x.lower())
ignore_idxs = []
for i in range(10000):
    if 'the' not in df_covid_small.iloc[i]['filtered_body_text']:
        ignore_idxs.append(i)
for i in ignore_idxs[:10]:
    print (df_covid_small.iloc[i]['filtered_body_text'][:250]+"......")
    print("="*140)
count = 0
for i in range(10000):
    if len(df_covid_small.iloc[i]['filtered_body_text']) < 500:
        if i not in ignore_idxs:
            if count < 10:
                print(df_covid_small.iloc[i]['filtered_body_text']+"....")
                print("="*140)
            ignore_idxs.append(i)
            count += 1
df_covid_small.drop(df_covid_small.index[ignore_idxs], inplace=True)
customize_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',
    '-PRON-'
]
final_stop_words = set(stopwords.words('english')).union(STOP_WORDS).union(set(customize_stop_words))
#From coursera course on sequence models
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map

root_path = '/kaggle/input/glove6b'
word_to_vec_map = read_glove_vecs(f'{root_path}/glove.6B.100d.txt')
doc_embeddings = np.zeros((len(df_covid_small),100))
def words_freq_atleast_2(filtered_body_text):
    #tokenize the text
    tokens = nltk.word_tokenize(filtered_body_text)
    #remove stop words
    tokens = list(filter(lambda x: x not in final_stop_words, tokens))
    #lemmatize the words so that other forms of same word becomes a single word
    tokens = list(map(lambda x: nltk.WordNetLemmatizer().lemmatize(x), tokens))
    #return words which have frequency greater than 1
    return list(filter(lambda x: x[1] != 1, Counter(tokens).most_common()))

words_freq_atleast_2(df_covid_small.iloc[1]['filtered_body_text'])[:5]
norms_doc_embeddings = np.zeros((len(df_covid_small),1))
for i in range(len(df_covid_small)):
    if i % 1000 == 0:
        print("working on "+str(i)+"th document")
    words_freq = words_freq_atleast_2(df_covid_small.iloc[i]['filtered_body_text'])
    doc_embedding_vec = np.zeros(word_to_vec_map["a"].shape)
    num_words = 0
    for word_freq in words_freq:
        word = word_freq[0]
        freq = word_freq[1]
        try:
            #adding word embeddings for each word in the document
            doc_embedding_vec += (word_to_vec_map[word] * freq)
            num_words += freq
        except:
            continue
    try:
        # doing average
        doc_embedding_vec /= num_words
    except:
        print("divide by zero encountered for article at index "+str(i))
        continue
    norms_doc_embeddings[i,:] = np.sqrt(np.dot(doc_embedding_vec,doc_embedding_vec))
    doc_embeddings[i,:] = doc_embedding_vec
def get_kw_embedding_vec(key_words):
    #key_words=['virus', 'genetics', 'origin', 'evolution', 'real-time', 'tracking', 'whole', 'genomes']
    kw_embedding_vec = np.zeros(word_to_vec_map["a"].shape)
    for kw in key_words:
        try:
            kw_embedding_vec += word_to_vec_map[kw]
        except:
            continue
    kw_embedding_vec /= len(key_words)
    return kw_embedding_vec
def get_kw_from_question(question):
    question = question.lower()
    #remove hyphen from string.punctuation and remove all other punctuation
    question = question.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
    tokens = nltk.word_tokenize(question)
    #remove stop words
    tokens = list(filter(lambda x: x not in final_stop_words, tokens))
    #lemmatize the words so that other forms of same word becomes a single word
    tokens = list(map(lambda x: nltk.WordNetLemmatizer().lemmatize(x), tokens))
    #return words which have frequency greater than 1
    return list(set(tokens))
question = "What do we know about virus genetics, origin, and evolution? What do we know about Real-time tracking of whole genomes"
keywords = get_kw_from_question(question)
keywords
kw_embedding_vec = get_kw_embedding_vec(keywords)
norm_kw_embedding = np.sqrt(np.dot(kw_embedding_vec, kw_embedding_vec))
norm_prods = norms_doc_embeddings * norm_kw_embedding
kw_embedding_vec = kw_embedding_vec.reshape(100,1)
dot_prod = np.dot(doc_embeddings, kw_embedding_vec)
cosine_similarity = dot_prod / norm_prods
fig = plt.figure(figsize=(20,10))
plt.plot(cosine_similarity[:100])
plt.xlabel('document/article index')
plt.ylabel('cosine similarity with question keywords')
cosine_similarity = cosine_similarity.reshape(-1,)
reqd_idxs = np.argpartition(cosine_similarity, -5)[-5:]
cosine_similarity[reqd_idxs]
reqd_idxs = reqd_idxs[np.argsort(cosine_similarity[reqd_idxs])][::-1]
reqd_idxs
print("Articles for your question: ")
for idx in reqd_idxs:
    try:
        print(meta_df.loc[meta_df['sha'] == df_covid_small.iloc[idx]['paper_id']]['url'].iloc[0])
    except:
        print("url not found")
    print(df_covid_small.iloc[idx]['title']+" by "+ df_covid_small.iloc[idx]['authors'])
    print(df_covid_small.iloc[idx]['body_text'][:1000]+"......")
    print()
def compute_cosine_get_idxs(norms_doc_embeddings, norm_kw_embedding, kw_embedding_vec, doc_embeddings):
    norm_prods = norms_doc_embeddings * norm_kw_embedding
    kw_embedding_vec = kw_embedding_vec.reshape(100,1)
    dot_prod = np.dot(doc_embeddings, kw_embedding_vec)
    cosine_similarity = dot_prod / norm_prods
    cosine_similarity = cosine_similarity.reshape(-1,)
    reqd_idxs = np.argpartition(cosine_similarity, -5)[-5:]
    reqd_idxs = reqd_idxs[np.argsort(cosine_similarity[reqd_idxs])][::-1]
    return reqd_idxs
question = "What is the evidence that livestock could be infected (field surveillance, genetic sequencing, receptor binding)? \
Evidence of whether farmers are infected, and whether farmers could have played a role in the origin."
keywords = get_kw_from_question(question)
kw_embedding_vec = get_kw_embedding_vec(keywords)
norm_kw_embedding = np.sqrt(np.dot(kw_embedding_vec, kw_embedding_vec))
reqd_idxs = compute_cosine_get_idxs(norms_doc_embeddings, norm_kw_embedding, kw_embedding_vec, doc_embeddings)
print("Articles for your question: ")
for idx in reqd_idxs:
    try:
        print(meta_df.loc[meta_df['sha'] == df_covid_small.iloc[idx]['paper_id']]['url'].iloc[0])
    except:
        print("url not found")
    print(df_covid_small.iloc[idx]['title']+" by "+ df_covid_small.iloc[idx]['authors'])
    print(df_covid_small.iloc[idx]['body_text'][:1000]+"......")
    print()
question = "What about Animal host and any evidence of continued spill over to humans, Socioeconomic and behavioral risk factors for this spill over?"
keywords = get_kw_from_question(question)
kw_embedding_vec = get_kw_embedding_vec(keywords)
norm_kw_embedding = np.sqrt(np.dot(kw_embedding_vec, kw_embedding_vec))
reqd_idxs = compute_cosine_get_idxs(norms_doc_embeddings, norm_kw_embedding, kw_embedding_vec, doc_embeddings)
print("Articles for your question: ")
for idx in reqd_idxs:
    try:
        print(meta_df.loc[meta_df['sha'] == df_covid_small.iloc[idx]['paper_id']]['url'].iloc[0])
    except:
        print("url not found")
    print(df_covid_small.iloc[idx]['title']+" by "+ df_covid_small.iloc[idx]['authors'])
    print(df_covid_small.iloc[idx]['body_text'][:1000]+"......")
    print()
question = "What are the sustainable risk reduction strategies?"
keywords = get_kw_from_question(question)
kw_embedding_vec = get_kw_embedding_vec(keywords)
norm_kw_embedding = np.sqrt(np.dot(kw_embedding_vec, kw_embedding_vec))
reqd_idxs = compute_cosine_get_idxs(norms_doc_embeddings, norm_kw_embedding, kw_embedding_vec, doc_embeddings)
print("Articles for your question: ")
for idx in reqd_idxs:
    try:
        print(meta_df.loc[meta_df['sha'] == df_covid_small.iloc[idx]['paper_id']]['url'].iloc[0])
    except:
        print("url not found")
    print(df_covid_small.iloc[idx]['title']+" by "+ df_covid_small.iloc[idx]['authors'])
    print(df_covid_small.iloc[idx]['body_text'][:1000]+"......")
    print()