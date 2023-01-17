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
import pandas as pd 



from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

import tensorflow as tf

from tensorflow.keras.layers import Dropout

from nltk.stem.lancaster import LancasterStemmer

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

import matplotlib.pylab as plt



import nltk

import re

from nltk.corpus import stopwords
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

train_df.head()
# Nan value dropping

train_df = train_df.dropna()



# train_data_X

X = train_df.drop('sentiment', axis = 1)



#  train_data_y as target

y = train_df['sentiment']
# stopwords download

nltk.download('stopwords')
# for further processing

temp_data = X.copy()



# chagein original index to number

temp_data.reset_index(inplace = True)
temp_data.head()
len_result = [len(s) for s in temp_data]
print(len_result)
unique_elements, counts_elements = np.unique(y, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
plt.subplot(1,2,1)

plt.bar(['neg','neu','pos'],counts_elements )



plt.show()
# Preprocessing :stemer : LancasterStemmer



stemmer = LancasterStemmer()

corpus = []



for i in range(0, len(temp_data)):

    

    result = re.sub('[^a-zA-Z]', ' ', str(temp_data['text'][i]))

    result = result.lower()

    result = result.split()

    

    result = [stemmer.stem(word) for word in result if not word in stopwords.words('english')]

    result = ' '.join(result)

    corpus.append(result)
# voc size

voc_size = 5000
# One Hot Encoding

onehot_code = [one_hot(words, voc_size) for words in corpus]
# making same length sentences

sent_length = 30

embedded_size = pad_sequences(onehot_code, padding = 'pre', maxlen = sent_length)
# Finding the numberof labels

text_num = len(temp_data['text'])

num_labels = len(set(train_df['sentiment']))

num_labels_set = set(train_df['sentiment'])



print(text_num,num_labels,num_labels_set)
# structure setting

vector_size = 30 



## Creating model

model=Sequential()

model.add(Embedding(voc_size,vector_size,input_length=sent_length))

model.add(LSTM(100))#100

model.add(Dense(num_labels,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

print(model.summary())
from sklearn import preprocessing



# labeel encoding 

label_enconding = preprocessing.LabelEncoder()

y = label_enconding.fit_transform(y)



X_final = np.array(embedded_size)

y_final = np.array(y)



from keras.utils import to_categorical

y_final = to_categorical(y_final)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 20, batch_size = 64)
def preprocess(X):

    

    # Drop Nan Values

    X = X.fillna(0)

    

    temp_data = X.copy()



    temp_data.reset_index(inplace = True)



    # Dataset Preprocessing

    stemmer = LancasterStemmer()



    corpus = []



    for i in range(0, len(temp_data)):

        

        result = re.sub('[^a-zA-Z]', ' ', str(temp_data['text'][i]))

        result = result.lower()

        result = result.split()



        result = [stemmer.stem(word) for word in result if not word in stopwords.words('english')]

        result = ' '.join(result)

        corpus.append(result)



    # voc size

    voc_size = 5000



    onehot_code = [one_hot(words, voc_size) for words in corpus]



    

    # making  same length sentence

    sent_length = 30

    

    # Embedding Representation

    embedded_size = pad_sequences(onehot_code, padding = 'pre', maxlen = sent_length)



    X_final = np.array(embedded_size)

    

    

    return X_final, X
# reading test data and pre-processing

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

X_test,X_test_drop = preprocess(test_df)
y_pred_test = model.predict_classes(X_test)
submission_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
submission_data.head()

df_sub = pd.DataFrame()

df_sub['id'] = X_test_drop['textID']

df_sub['text'] = X_test_drop['text']

df_sub['sentiment_predicted'] = label_enconding.inverse_transform(y_pred_test)

df_sub['sentiment_actual'] = X_test_drop['sentiment']
df_sub.to_csv('submission.csv', index=False)
df_sub.head()
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve,auc

import seaborn as sns

import matplotlib.pyplot as plt



cm = confusion_matrix(df_sub['sentiment_actual'].values , df_sub['sentiment_predicted'].values)

cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
#Transform to df for easier plotting

final_cm = pd.DataFrame(cm, index = label_enconding.classes_,

                     columns = label_enconding.classes_

                    )
plt.figure(figsize = (5,5))

sns.heatmap(final_cm, annot = True,cmap='Greys',cbar=False)

plt.title('Emotion Classify')

plt.ylabel('True class')

plt.xlabel('Prediction class')

plt.show()