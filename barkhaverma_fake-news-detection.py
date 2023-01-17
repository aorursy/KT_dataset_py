import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
Train = pd.read_csv("../input/fake-news/train.csv")

# here we are printing first five lines of our train dataset
Train.head()
# here we are Getting the Independent Features
X=Train.drop('label',axis=1)
# printing head of our independent features
X.head()
# here we are printing shape of our dataset
Train.shape

# here we are checking if there is null value or not
Train.isnull().sum()
# here we are droping NaN values from our dataset
Train=Train.dropna()
# here we are checking again if there is any NaN value or not
Train.isnull().sum()
Train.head(10)
# here we are copying our dataset .
Train=Train.copy()
# here we are reseting our index
Train.reset_index(inplace=True)
# here we are printing our first 10 line of dataset for checking indexing
Train.head(10)
x=Train['title']
# here we are making independent features
y=Train['label']
y.shape
# here we are importing nltk,stopwords and porterstemmer we are using stemming on the text 
# we have and stopwords will help in removing the stopwords in the text

#re is regular expressions used for identifying only words in the text and ignoring anything else
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(Train)):
    review = re.sub('[^a-zA-Z]', ' ', Train['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus[30]
# here we are setting vocabulary size
voc_size=5000
# here we are performing one hot representation
from tensorflow.keras.preprocessing.text import one_hot
one_hot_rep=[one_hot(words,voc_size)for words in corpus] 
# here we are printing length of first line
len(one_hot_rep[0])
# here we are printing length of 70 line
len(one_hot_rep[70])
# here we are importing library for doind padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
# here we are specifying a sentence length so that every sentence in the corpus will be of same length

sentence_length=25

# here we are using padding for creating equal length sentences


embedded_docs=pad_sequences(one_hot_rep,padding='pre',maxlen=sentence_length)
print(embedded_docs)
# here we are imporitng important libraries for building model
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
 
#Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sentence_length))
model.add(Dropout(0.3))
model.add(LSTM(200))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
z =np.array(embedded_docs)
y =np.array(y)
# here we are printing shape 
z.shape,y.shape
# here we are splitting the data for training and testing the model

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(z, y, test_size=0.10, random_state=42)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,batch_size=64)
# here we are loading our test dataset for prediction

Test=pd.read_csv('../input/fake-news/test.csv') 
Test_id=Test["id"]
Test_id
# here we are printing first 5 line of our dataset
Test.head()
# here we are removing these columns as they are not so important
Test=Test.drop(['text','id','author'],axis=1)
# printing first 5 line of our dataset
Test.head()
# here we are checking if null values in the test dataset or not

Test.isnull().sum()
Test.fillna('fake fake fake',inplace=True)
# here we are filling NaN value with "fake,fake,fake".we cannot drop the NaN value because
# as the solution file that we have to submitted in kaggle expects 
# it to have 5200 rows so we can't drop rows in the test dataset
# here we are printing shape
Test.shape
# here we are creating corpus for the test dataset exactly the same as we created for the 
# training dataset
ps = PorterStemmer()
corpus_test = []
for i in range(0, len(Test)):
    review = re.sub('[^a-zA-Z]', ' ',Test['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)
corpus[30]
# here we are creating one hot representation for the test corpus

one_hot_rep_Test=[one_hot(words,voc_size)for words in corpus_test] 
# here we are doing padding for the test dataset
sentence_length=25

embedded_docs_test=pad_sequences(one_hot_rep_Test,padding='pre',maxlen=sentence_length)
print(embedded_docs_test)
x_test=np.array(embedded_docs_test)
#making predictions for the test dataset

check=model.predict_classes(x_test)
check
check.shape
Test.shape
val=[]
for i in check:
    val.append(i[0])
submission = pd.DataFrame({'id':Test_id, 'label':val})
submission.shape
submission.head()
#saving the submission file

submit_sample.to_csv('submission.csv',index=False)