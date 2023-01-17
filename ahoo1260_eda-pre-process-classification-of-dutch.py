import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential,Model

from keras import optimizers

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D,Input,Concatenate

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from sklearn.cluster import AgglomerativeClustering



stop=set(stopwords.words('dutch'))

df=pd.read_excel("/kaggle/input/pacmed/triage_example.xlsx")
df.columns
df.head(5)
x=df['Urgency indication'].value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
augmented_row={"Ingangsklacht":["griep"] ,"Triage: H":["Wat te doen?"] ,"Triage: B":["plotselinge koorts, pijntjes, zwakte of verlies van eetlust. In het bijzonder samen hoesten en koorts hebben"] , "Triage: M":["Geen"],"Triage: V": ["Geen"],"Triage: Checklist":["Kortademig: Nee; Blaarvormig: Nee; Zieke indruk: Nee; Ontsteking: Nee"],"Age": [23],"Gender": ["Female"], "Urgency indication":["Non-urgent"]}
df_augmented = pd.DataFrame(augmented_row, columns = list(df.columns))
df=pd.concat([df,df_augmented])
df
x=df['Urgency indication'].value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
X_train=df.iloc[:, :-1]

y_train=df.iloc[:,-1]
y_train= pd.Categorical(y_train).codes
y_train
X_train
categorical_features=X_train.drop(['Triage: B'],axis=1)

text_feature=X_train['Triage: B']
categorical_features
len(categorical_features)
Kortademig=[]

Blaarvormig=[]

indruk=[]

Ontsteking=[]

for i in range(len(categorical_features)):

    checklist_text=categorical_features.iloc[i]['Triage: Checklist']

    items=checklist_text.split(';')

    Kortademig.append(items[0].split(": ",1)[1])

    Blaarvormig.append(items[1].split(": ",1)[1])

    indruk.append(items[2].split(": ",1)[1])

    Ontsteking.append(items[3].split(": ",1)[1])

        
categorical_features['Kortademig']=Kortademig

categorical_features['Blaarvormig']=Blaarvormig

categorical_features['indruk']=indruk

categorical_features['Ontsteking']=Ontsteking

categorical_features=categorical_features.drop(['Triage: Checklist'],axis=1)

categorical_features
categorical_features['Ingangsklacht'] = pd.Categorical(categorical_features['Ingangsklacht']).codes

categorical_features['Triage: H'] = pd.Categorical(categorical_features['Triage: H']).codes

categorical_features['Triage: M'] = pd.Categorical(categorical_features['Triage: M']).codes

categorical_features['Triage: V'] = pd.Categorical(categorical_features['Triage: V']).codes

categorical_features['Gender'] = pd.Categorical(categorical_features['Gender']).codes

categorical_features['Kortademig'] = pd.Categorical(categorical_features['Kortademig']).codes

categorical_features['Blaarvormig'] = pd.Categorical(categorical_features['Blaarvormig']).codes

categorical_features['indruk'] = pd.Categorical(categorical_features['indruk']).codes

categorical_features['Ontsteking'] = pd.Categorical(categorical_features['Ontsteking']).codes

categorical_features
#put all documents together

corpus=text_feature.str.cat(sep=' ,')
corpus
#get the words

words=corpus.split()
words
dic=defaultdict(int)

for word in words:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)

plt.bar(x,y)


dic=defaultdict(int)

import string

special = string.punctuation

for word in words:

    for ch in word:

        if ch in special:

            dic[ch]+=1

        



x,y=zip(*dic.items())

plt.bar(x,y)
counter=Counter(words)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
sns.barplot(x=y,y=x)

text_feature=text_feature.str.lower()

text_feature
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
text_feature=text_feature.apply(lambda x : remove_punct(x))

text_feature
def clean_stopwords(text):

    pattern = re.compile(r'\b(' + r'|'.join(stop) + r')\b\s*')

    text = pattern.sub(' ', text)

    return text
text_feature=text_feature.apply(lambda x : clean_stopwords(x))

text_feature
def get_top_text_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_text_bigrams=get_top_text_bigrams(text_feature)[:10]

x,y=map(list,zip(*top_text_bigrams))

sns.barplot(x=y,y=x)
def get_top_text_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_text_trigrams=get_top_text_trigram(text_feature)[:10]

x,y=map(list,zip(*top_text_trigrams))

sns.barplot(x=y,y=x)
vectorizer = CountVectorizer(ngram_range=(1,4))

text_feature_vector= vectorizer.fit_transform(text_feature)

print(vectorizer.get_feature_names())
text_feature_vector
text_feature_vector=text_feature_vector.todense()
#converting matrix to numpy array

text_feature_vector=np.array(text_feature_vector)
#converting dataframe to numpy array

categorical_features=categorical_features.values
final_features_vectors=[]

for i in range(len(categorical_features)):

    vector=np.concatenate((categorical_features, text_feature_vector), axis=None)

    final_features_vectors.append(vector)
final_features_vectors
from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(final_features_vectors, y_train)
predictions=clf.predict(final_features_vectors)

predictions
def get_accuracy(predictions,realValues):

    correct=0

    incorrect=0

    for i in range(len(predictions)):

        if predictions[i]==realValues[i]:

            correct=correct+1

        else:

            incorrect=incorrect+1

    return correct/(correct+incorrect)



        
get_accuracy(predictions,y_train)
# please un-comment these lines to download fasttext model:

# import urllib.request

# urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz", "cc.nl.300.vec")
embedding_dict={}

with open('/kaggle/input/fasttext-dutch/cc.nl.300.vec','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(text_feature)

sequences=tokenizer_obj.texts_to_sequences(text_feature)



text_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
found=0

not_found=0

num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,300))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec

        found=found+1

    else:

        not_found=not_found+1
print("number of words which found a vector from vocabulary is: ",found)

print("number of words which haven't found a vector from vocabulary is: ",not_found)
model=Sequential()

embedding=Embedding(num_words, 300,

          weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=optimizers.Adam(learning_rate=1e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])



model.summary()
model.fit(text_pad,y_train)
predictions=model.predict(text_pad)
def get_binary_output(val):

    if val>=0.5:

        return 1

    else:

        return 0
predictions=[get_binary_output(val) for val in predictions]
predictions
get_accuracy(predictions, y_train)
nlp_input=Input(shape=(MAX_LEN,),name='nlp_input')

categorical_input = Input((10,))





embedding=Embedding(input_dim=num_words,output_dim=300,weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)(nlp_input)

drop_layer=SpatialDropout1D(0.2)(embedding)

Lstm_layer = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(drop_layer)



# Concatenate the convolutional features and the vector input

concat_layer= Concatenate()([categorical_input, Lstm_layer])

output = Dense(1, activation='sigmoid')(concat_layer)



# define a model with a list of two inputs

model = Model(inputs=[nlp_input, categorical_input], outputs=output)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

model.summary()
model.fit([text_pad,categorical_features],y_train)
predictions=model.predict([text_pad,categorical_features])
predictions
predictions=[get_binary_output(val) for val in predictions]
get_accuracy(predictions, y_train)
def get_Hierarchical_Clusters(words_vectors,NUMBER_OF_CLUSTERS):



    cluster = AgglomerativeClustering(n_clusters=NUMBER_OF_CLUSTERS, affinity='euclidean', linkage='ward')

#     fit_predict fits the hierarchical clustering from features, and return cluster labels.

    cluster_labels= cluster.fit_predict(words_vectors)

    return cluster_labels
clusters = get_Hierarchical_Clusters(final_features_vectors, 2)

clusters
from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.feature_extraction.text import CountVectorizer



number_words=5



def lda(data, NUMBER_OF_CLUSTERS):

    count_vectorizer = CountVectorizer(stop)

    count_data = count_vectorizer.fit_transform(data)



    lda = LDA(n_components=NUMBER_OF_CLUSTERS)

    lda.fit(count_data)

    

    all_topics_words=[]

    words=count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(lda.components_):

        each_topic_words=([words[i] for i in topic.argsort()[:-number_words - 1:-1]])

        # each_topic_words=([words[i] for i  in topic.argsort()[::-1]])

        all_topics_words.append(each_topic_words)



    return lda, all_topics_words
NUMBER_OF_CLUSTERS=2

lda, topic_words=lda(text_feature,NUMBER_OF_CLUSTERS)

#printing topic words:

for i in range(NUMBER_OF_CLUSTERS):

    print("topic "+str(i)+": "+'[%s]' % ', '.join(map(str, topic_words[i])))
#clustering based on the lda model:

count_vectorizer = CountVectorizer(stop)

count_data = count_vectorizer.fit_transform(text_feature)

lda.transform(count_data)