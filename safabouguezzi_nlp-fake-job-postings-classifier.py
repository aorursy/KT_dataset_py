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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf
data = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
data.head(2)
data.info()
data.drop(['department','salary_range','job_id'], axis=1, inplace=True)
data.fillna(" ", inplace=True)
data.head(2)
data['job_description'] = data['title'] + ' ' + data['location'] + ' ' + data['company_profile'] + ' ' + data['description'] + ' ' + data['requirements'] + ' ' + data['benefits'] + ' ' + data['employment_type'] + ' ' + data['required_experience'] + ' ' + data['required_education'] + ' ' + data['industry'] + ' ' + data['function'] 
data.drop(['title','location','company_profile','description','requirements','benefits','employment_type','required_experience','required_education','industry','function'], axis = 1, inplace= True)
data.head(2)
data.reset_index(inplace= True)
data['fraudulent'].hist(figsize=(10,5))
data['fraudulent'].value_counts()
import nltk

import re , string

from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

punctuations = set(string.punctuation)

stop_words.update(punctuations)
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

# text = []

# for i in range(0, len(data['job_description'])):

#     clean_text = re.sub('[^a-zA-Z]', ' ', data['job_description'][i])

#     clean_text = clean_text.lower().split()    

#     clean_text = [lemmatizer.lemmatize(word) for word in clean_text if not word in stop_words]

#     clean_text = ' '.join(clean_text)

#     text.append(clean_text)
from nltk.stem.snowball import SnowballStemmer

lemmatizer = SnowballStemmer(language='english')

text = []

for i in range(0, len(data['job_description'])):

    clean_text = re.sub('[^a-zA-Z]', ' ', data['job_description'][i])

    clean_text = clean_text.lower().split()

    clean_text = [lemmatizer.stem(word) for word in clean_text if not word in stop_words]

    clean_text = ' '.join(clean_text)

    text.append(clean_text)
text[0]


from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

voc_size=5000
onehot_text=[one_hot(words,voc_size)for words in text] 

onehot_text[0]
sent_length=100

sent_with_same_lenght = pad_sequences(onehot_text,padding='post',maxlen=sent_length)

print(sent_with_same_lenght)
sent_with_same_lenght[0]
#  model

embedding_vector_features=30 #80

model1=Sequential()

model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model1.add(Bidirectional(LSTM(150))) #combined two LSTM: one works from the start to the end, the second works from the end to the start  #90

model1.add(Dropout(0.40))

model1.add(Dense(1,activation='sigmoid')) 

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model1.summary())
len(sent_with_same_lenght),data['fraudulent'].shape
import numpy as np

X_final=np.array(sent_with_same_lenght)

y_final=np.array(data['fraudulent'])
X_final[0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42) 

model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=15,batch_size=64) 
y_pred=model1.predict_classes(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))