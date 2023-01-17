# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

import string

import re

from string import digits

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

from matplotlib import pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("\nTrain data: \n",train.head())

print("\nTest data: \n",test.head())
train_data=train.drop(train.columns[0], axis=1) 

test_data=test

print(train_data.head())

print(test_data.head())
train_comments=train_data.iloc[:,0]

test_comments=test_data.iloc[:,1]



#saving index to separate them later

train_comments_index=train_comments.index

test_comments_index=test_comments.index



frames = [train_comments, test_comments]

comments = pd.concat(frames, ignore_index=True)





labels=train_data.iloc[:,1:]



print("Train Comments Shape: ",train_comments.shape)

print("Test Comments Shape: ",test_comments.shape)

print("Comments Shape after Merge: ",comments.shape)

print("Comments are: \n",comments.head())

print("\nLabels are: \n", labels.head())
c=comments.str.translate(str.maketrans(' ', ' ', string.punctuation))

c.head()
c=c.str.translate(str.maketrans(' ', ' ', '\n'))

c=c.str.translate(str.maketrans(' ', ' ', digits))

c.head()
c=c.apply(lambda tweet: re.sub(r'([a-z])([A-Z])',r'\1 \2',tweet))

c.head()
c=c.str.lower()

c.head()
c=c.str.split()

c.head()
stop = set(stopwords.words('english'))

c=c.apply(lambda x: [item for item in x if item not in stop])

c.head()    
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()

com=[]

for y in tqdm(c):

    new=[]

    for x in y:

        z=lemmatizer.lemmatize(x)

        z=lemmatizer.lemmatize(z,'v')

        new.append(z)

    y=new

    com.append(y)
clean_data=pd.DataFrame(np.array(com), index=comments.index,columns={'comment_text'})

clean_data['comment_text']=clean_data['comment_text'].str.join(" ")

print(clean_data.head())

train_clean_data=clean_data.loc[train_comments_index]

test_clean_data=clean_data.drop(train_comments_index,axis=0).reset_index(drop=True)

print("PreProcessed Train Data : ",train_clean_data.head(5))

print("PreProcessed Test Data : ",test_clean_data.head(5))

frames=[train_clean_data,labels]

train_result = pd.concat(frames,axis=1)

frames=[test.iloc[:,0],test_clean_data]

test_result = pd.concat(frames,axis=1)

print(train_result.head())

print(test_result.head())
temp_df=train_result.iloc[:,2:-1]

corr=temp_df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True)
tf_idf = TfidfVectorizer(max_features=50000, min_df=2)

tfidf_train = tf_idf.fit_transform(train_result['comment_text'])

tfidf_test = tf_idf.transform(test_result['comment_text'])

# import pickle

# pickle.dump(tf_idf.vocabulary_,open("feature.pkl","wb"))
from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

model = Sequential()

model.add(Dense(100,activation='relu',input_shape=(50000,)))

model.add(Dense(100,activation='relu',input_shape=(50000,)))

model.add(Dense(6,activation='sigmoid'))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

model.fit(tfidf_train, train_result[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values)

y_pred = model.predict(tfidf_test)
dict = {

    'id': test_result.id.values,

    'toxic' : y_pred[:,0],

    'severe_toxic' : y_pred[:,1],

    'obscene':y_pred[:,2],

    'threat':y_pred[:,3],

    'insult':y_pred[:,4],

    'identity_hate':y_pred[:,5]

}

ans = pd.DataFrame(dict)

ans

ans.to_csv('Submit1.csv',index=False)
s = input()

c = s.translate(str.maketrans(' ', ' ', string.punctuation))

c = c.translate(str.maketrans(' ', ' ', '\n'))

c = c.translate(str.maketrans(' ', ' ', digits))

c = re.sub(r'([a-z])([A-Z])', r'\1 \2', c)

c = c.lower()

c = c.split()

stop = set(stopwords.words('english'))

c = [item for item in c if item not in stop]

from tqdm import tqdm

lemmatizer = WordNetLemmatizer()

com = []

for y in tqdm(c):

    new = []

    for x in y:

        z = lemmatizer.lemmatize(x)

        z = lemmatizer.lemmatize(z, 'v')

        new.append(z)

    y = new

    com.append(y)

clean = ""

for i in com:

    t = ''

    clean += t.join(i) + " "

test = tf_idf.transform(np.array([clean]))

y_pred = model.predict(test)

pred = pd.DataFrame(

{

    'label':labels.columns,

    'probability':y_pred[0]

})

# print(train.columns)

print(pred)
