import numpy as np
import pandas as pd 
import os
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
import glob
import json
all_json = glob.glob(f'{root_path}/custom_license/custom_license/pdf_json/*.json', recursive=True)
len(all_json)

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)
def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json[:5]):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
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
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()
import nltk
from nltk.corpus import stopwords
#paragraph tokenization, figured 6 lines was a rough estimate of a paragraph 
bodies = df_covid["body_text"]
sentences = []
b=0
s=0
lines = 0
stopWords = set(stopwords.words('english'))
stopWords.add('(')
stopWords.add(')')
stopWords.add(',')
stopWords.add(', ')

for b in range(len(bodies)):
    tokens = nltk.word_tokenize(bodies[b])
    words = []
    for t in range(len(tokens)):
        if tokens[t] == '.':
            words.append(tokens[t])
            lines += 1
            if lines > 6:
                sentences.append(words)
                words = []
                lines = 0
        elif tokens[t] not in stopWords:
            words.append(tokens[t])
            
sentences[3:4]
from gensim.models import Word2Vec
import nltk
import numpy as np 
from sklearn import metrics
 
model = Word2Vec(sentences, min_count=1)
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
X=[]
for sentence in sentences:
    X.append(sent_vectorizer(sentence, model))   

print (model.similarity('infectious', 'virus'))
print(model.most_similar(positive=['virus', 'illness', 'symptom'], negative=[], topn=10)) 
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
from sklearn import cluster
from sklearn import metrics
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
stopWords.add('(')
stopWords.add(')')
stopWords.add(',')
stopWords.add(', ')
search = "virus infectious"
prepare_search = nltk.word_tokenize(search)
search_tokens = []
for w in prepare_search:
    if w not in stopWords:
        search_tokens.append(w)
similar_tokens = model.most_similar(positive=search_tokens, negative=[], topn=len(search_tokens))
for s in similar_tokens:
    search_tokens.append(s[0])
search_tokens
i=0
for i in range(len(df_covid)):
    for x in search_tokens:
        if df_covid['body_text'][i].find(x) > 0:
            print(df_covid['title'][i])
            print("===========================================================================================================================")
            break;
import smart_open
import gensim
def read_corpus(fname, tokens_only=False):
    for i, line in enumerate(fname):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
                
train_corpus = list(read_corpus(df_covid['body_text']))
train_corpus[3]
doc_model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=1)
doc_model.build_vocab(train_corpus)
doc_model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
ranks = []
second_ranks = []
text = "the increasing in the world that has resulted from globalization has significant implications for nursing and healthcare the american academy of nursing"
for doc_id in range(len(train_corpus)):
    inferred_vector = doc_model.infer_vector([text])
    sims = doc_model.docvecs.most_similar([inferred_vector], topn=len(doc_model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

    second_ranks.append(sims[1])
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(nltk.word_tokenize(text))))
sims = doc_model.docvecs.most_similar([inferred_vector], topn=len(doc_model.docvecs))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % doc_model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
from keras.utils import to_categorical
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
data = df_covid.drop(['paper_id', 'body_text', 'authors', 'journal','abstract_summary'], axis=1)
data = data.dropna().reset_index(drop=True)[:10]
print(len(data))
#single word tokenization
stopWords = set(stopwords.words('english'))
stopWords.add('(')
stopWords.add(')')
stopWords.add(',')
stopWords.add(', ')
proc_data = pd.DataFrame(columns = ['abstract', 'title'])

for x in range (len(data)):
    tokens = nltk.word_tokenize(data['abstract'][x])
    #words = []
    for t in range(len(tokens)):
        if tokens[t] not in stopWords:
            proc_data = proc_data.append({'abstract' : tokens[t] , 'title' : data['title'][x]} , ignore_index=True)
#sentence tokenization 
stopWords = set(stopwords.words('english'))
stopWords.add('(')
stopWords.add(')')
stopWords.add(',')
stopWords.add(', ')
proc_data = pd.DataFrame(columns = ['abstract', 'title'])

for x in range (len(data)):
    tokens = nltk.word_tokenize(data['abstract'][x])
    words = []
    for t in range(len(tokens)):
        if tokens[t] not in stopWords:
            if tokens[t] == '.':
                proc_data = proc_data.append({'abstract' : words , 'title' : data['title'][x]} , ignore_index=True)
                words = []
            else:
                words.append(tokens[t])
proc_data = proc_data.reset_index(drop=True)
proc_data.tail()
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedShuffleSplit

label_encoder = preprocessing.LabelEncoder() 
labels = label_encoder.fit_transform(proc_data['title'])
abstracts = label_encoder.fit_transform(proc_data['abstract'])

#stratified shuffle split for testing of the model. 
#We put all words into a list according to the paper it came from as label

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss.get_n_splits(abstracts, labels)

for train_index, test_index in sss.split(abstracts, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_x, test_x = abstracts[train_index], abstracts[test_index]
    train_y, test_y = labels[train_index], labels[test_index]
print(test_x[0])
print(test_y[0])
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam
from keras.utils import plot_model
def get_model():
    lstm = Sequential()

    # Embedding layer
    lstm.add(
        Embedding(
            input_dim=1,
            output_dim=90,
            weights=None,
            trainable=True))

    # Recurrent layer with return sequences disabled (not really reccurent)
    lstm.add(
        LSTM(
            60, return_sequences=False, dropout=0.1,
            recurrent_dropout=0.1))

    # Fully connected layer
    lstm.add(Dense(1, activation='relu'))

    # Dropout for regularization
    lstm.add(Dropout(0.1))

    # Output layer
    lstm.add(Dense(90, activation='softmax'))

    # Compiling the model
    lstm.compile(
        optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    lstm.summary()
    return lstm
lstm = get_model()
history = lstm.fit(train_x,  train_y, 
                    batch_size=16, epochs=10,
                    validation_data=(test_x, test_y))
sample_id = 0
dicts = np.unique(df_covid['title'])
preds = lstm.predict_classes(test_x[:10])
print("Predicted: ", dicts[preds[sample_id]])
print("Actual: ", dicts[test_y[sample_id]])

