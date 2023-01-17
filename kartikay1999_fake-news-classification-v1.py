# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import nltk

from nltk.corpus import stopwords

import string

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag

from nltk.tokenize import sent_tokenize,word_tokenize
lemmatizer=WordNetLemmatizer()

stops=set(stopwords.words('english'))

puncts=list(string.punctuation)

stops.update(puncts)
fake_df=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true_df=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake_df['Type']='Fake'

true_df['Type']='True'

fake_df.info()
from nltk.corpus import wordnet

def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    if tag.startswith('V'):

        return wordnet.VERB

    if tag.startswith('N'):

        return wordnet.NOUN

    if tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
def clean_review(words):

    output_words=[]

    words=word_tokenize(words)

    for w in words:

        if w.lower() not in stops:

            pos=pos_tag([w])

            clean_word=lemmatizer.lemmatize(w,pos=get_simple_pos(pos[0][1]))

            output_words.append(clean_word.lower())

    return output_words
data=pd.concat([fake_df,true_df])
data = data.sample(frac=1).reset_index(drop=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['Type_encoded']=le.fit_transform(data['Type'])
data.columns
len(data['text'][0])
from tqdm import tqdm
text_data=[]

for i in tqdm(range(len(data[:1000]))):

    text_data.append(clean_review(data['title'][i]+data['text'][i]))
data_proto=data[:1000]

data_proto['processed_text']=text_data

data_proto
training_documents=data_proto[['Type_encoded','processed_text']][:650]

testing_documents=data_proto[['Type_encoded','processed_text']][650:]
all_words=[]

for doc in training_documents['processed_text']:

    all_words+=doc
freq=nltk.FreqDist(all_words)
common=freq.most_common(1000)

features=[i[0] for i in common]
features
def get_feature_dict(words):

    current_features={}

    words_set=set(words)

    for w in features:

        current_features[w] =w in words_set

    return current_features
training_data=[]

for i in range(len(training_documents)):

    training_data.append((get_feature_dict(training_documents['processed_text'][i]),training_documents['Type_encoded'][i]))
testing_documents.reset_index(inplace=True,drop=True)
testing_data=[]

for i in range(len(testing_documents)):

    testing_data.append((get_feature_dict(testing_documents['processed_text'][i]),testing_documents['Type_encoded'][i]))
from nltk import NaiveBayesClassifier
classifier=NaiveBayesClassifier.train(training_data)
print("accuracy for naive bayes with only 1000 data points: ",nltk.classify.accuracy(classifier,testing_data))

classifier.show_most_informative_features(15)
from sklearn.svm import SVC

from nltk.classify.scikitlearn import SklearnClassifier
svc=SVC()

classifier_sklearn=SklearnClassifier(svc)
classifier_sklearn.train(training_data)
nltk.classify.accuracy(classifier_sklearn,testing_data)

print("accuracy for SVC with only 1000 data points: ",nltk.classify.accuracy(classifier_sklearn,testing_data))

w=clean_review(data['text'][1111])

d=get_feature_dict(w)
classifier.classify(d)
data.iloc[1111]