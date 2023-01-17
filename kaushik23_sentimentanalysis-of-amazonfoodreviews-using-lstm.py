# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)
import nltk



#preprocessing

from nltk.corpus import stopwords  #stopwords

from nltk import word_tokenize,sent_tokenize # tokenizing

from nltk.stem import PorterStemmer,LancasterStemmer  

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag



# for named entity recognition (NER)

from nltk import ne_chunk



# vectorizers for creating the document-term-matrix (DTM)

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer



# BeautifulSoup libraray

from bs4 import BeautifulSoup 



import re # regex



#model_selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#evaluation

from sklearn.metrics import accuracy_score,roc_auc_score 

from sklearn.metrics import classification_report

from mlxtend.plotting import plot_confusion_matrix


stop_words=set(nltk.corpus.stopwords.words('english'))
import warnings

warnings.filterwarnings("ignore")



import re

from bs4 import BeautifulSoup

from tqdm import tqdm

from nltk.stem import WordNetLemmatizer



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers
df_frame = pd.read_csv("../input/amazon-fine-food-reviews/Reviews.csv")
df_frame_copy = df_frame.copy()
df_frame_copy.head()
df_frame_copy=df_frame_copy[['Text','Score']]

df_frame_copy['review']=df_frame_copy['Text']

df_frame_copy['rating']=df_frame_copy['Score']

df_frame_copy.drop(['Text','Score'],axis=1,inplace=True)
print(df_frame_copy.shape)

df_frame_copy.head()
print(df_frame_copy['rating'].isnull().sum())

df_frame_copy['review'].isnull().sum()
df_frame_copy.drop_duplicates(subset=['rating','review'],keep='first',inplace=True)
print(df_frame_copy.shape)

df_frame_copy.head()
def label_sentiment(rating):

  if(rating<=3):

    return 0

  else:

    return 1
df_frame_copy['sentiment']=df_frame_copy['rating'].apply(label_sentiment)
df_frame_copy.drop(['rating'],axis=1,inplace=True)

df_frame_copy.head()

df_frame_copy['sentiment'].value_counts()
def decontract(text):

    text = re.sub(r"won\'t", "will not", text)

    text = re.sub(r"can\'t", "can not", text)

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)

    return text
lemmatizer = WordNetLemmatizer()

def preprocess_text(review):

    review = re.sub(r"http\S+", "", review)            

    review = BeautifulSoup(review, 'lxml').get_text()   

    review = decontract(review)                        

    review = re.sub("\S*\d\S*", "", review).strip()     

    review = re.sub('[^A-Za-z]+', ' ', review)          

    review = review.lower()                             

    review = [word for word in review.split(" ") if not word in stop_words]

    review = [lemmatizer.lemmatize(token, "v") for token in review] #Lemmatization

    review = " ".join(review)

    review.strip()

    return review

df_frame_copy['review'] = df_frame_copy['review'].apply(lambda x: preprocess_text(str(x)))

df_frame_copy['review'].head()
train_df, test_df = train_test_split(df_frame_copy, test_size = 0.2, random_state = 42)

print("Training data size : ", train_df.shape)

print("Test data size : ", test_df.shape)
top_words = 6000

tokenizer = Tokenizer(num_words=top_words)

tokenizer.fit_on_texts(train_df['review'])

list_tokenized_train = tokenizer.texts_to_sequences(train_df['review'])



max_review_length = 130

X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)

y_train = train_df['sentiment']
embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train,y_train, nb_epoch=3, batch_size=64, validation_split=0.2)
list_tokenized_test = tokenizer.texts_to_sequences(test_df['review'])

X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)

y_test = test_df['sentiment']

prediction = model.predict(X_test)

y_pred = (prediction > 0.5)

print("Accuracy of the model : ", accuracy_score(y_pred, y_test))

print('F1-score: ', f1_score(y_pred, y_test))
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('ROC- AUC Curve')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()