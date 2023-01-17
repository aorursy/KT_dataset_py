import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import matplotlib
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH='/kaggle/input/fake-and-real-news-dataset'

TRUE_FILE_PATH=os.path.join(PATH,'True.csv')

FAKE_FILE_PATH=os.path.join(PATH,'Fake.csv')
true_data_df=pd.read_csv(TRUE_FILE_PATH)

true_class=['True' for index in range(true_data_df.shape[0])]

fake_data_df=pd.read_csv(FAKE_FILE_PATH)

fake_class=['Fake' for index in range(fake_data_df.shape[0])]
labels=['True','Fake']

class_wise_counts=[true_data_df.shape[0],fake_data_df.shape[0]]
matplotlib.rcParams['figure.figsize']=(10,10)

plt.bar(labels,class_wise_counts,align='center', alpha=0.5,color='r')

plt.xlabel('Classes')

plt.ylabel('Counts')

plt.title('Count vs Classes')

plt.show()

print ("Ratio of fake is to real news:",(fake_data_df.shape[0]/true_data_df.shape[0]))
true_data_df['class']=true_class

fake_data_df['class']=fake_class
fake_data_df['class']=fake_class
true_data_df.head()
fake_data_df.head()
data_frame=pd.concat([true_data_df,fake_data_df],axis='rows')
data_frame.isnull().sum()
data_frame.head()
data_frame.date.value_counts()
data_frame.drop('date',axis='columns',inplace=True)
data_frame.head()
data_frame.subject.unique()
real_news_df=data_frame[data_frame.subject=='politicsNews']
real_news_df.shape
(fake_subject_keys,fake_counts)=np.unique(data_frame[data_frame['class']=='Fake'].subject,return_counts=True)

(true_subject_keys,true_counts)=np.unique(data_frame[data_frame['class']=='True'].subject,return_counts=True)
matplotlib.rcParams['figure.figsize']=(10,10)

plt.bar(fake_subject_keys,fake_counts,align='center', alpha=0.5,color='g')

plt.xlabel('Subjects')

plt.ylabel('Counts')

plt.title('FakeNewsCounts vs Subjects')

plt.show()
matplotlib.rcParams['figure.figsize']=(10,7)

plt.bar(true_subject_keys,true_counts,align='center', alpha=0.5,color='b')

plt.xlabel('Subjects')

plt.ylabel('Counts')

plt.title('TrueNewsCounts vs Subjects')

plt.show()
subject_dummies=pd.get_dummies(data_frame.subject)
data_frame2=pd.concat([data_frame,subject_dummies],axis='columns')
title_column=list(data_frame2.title)

text_column=list(data_frame2.text)
title_column[0]
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import string

import nltk

from nltk import pos_tag

from nltk.stem import WordNetLemmatizer
stop_words=stopwords.words('english')

stop_words.extend(string.punctuation)
from nltk.corpus import wordnet



def get_wordnet_pos(treebank_tag):



    if treebank_tag.startswith('J'):

        return wordnet.ADJ

    elif treebank_tag.startswith('V'):

        return wordnet.VERB

    elif treebank_tag.startswith('N'):

        return wordnet.NOUN

    elif treebank_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
lemmatizer=WordNetLemmatizer()



def clean_data(text):

    

    clean_words=[]

    words=word_tokenize(text)

    for word in words:

        if (word.lower() not in stop_words and word.isdigit()==False):

            curr_word_pos_tag=pos_tag([word])

            

            simple_pos_tag=get_wordnet_pos(curr_word_pos_tag[0][1])

            clean_words.append(lemmatizer.lemmatize(word,simple_pos_tag))

    return clean_words



clean_title_column=[clean_data(current_column) for current_column in title_column]

clean_title_column[0]
clean_text_column=[clean_data(current_column) for current_column in text_column]
clean_title_column_list=[" ".join(list_words) for list_words in clean_title_column]

clean_text_column_list=[" ".join(list_words) for list_words in clean_text_column]
data_frame2['title']=clean_title_column_list

data_frame2['text']=clean_text_column_list
from sklearn.utils import shuffle

data_frame3 = shuffle(data_frame2)
data_frame3.reset_index(inplace=True, drop=True)
train_dataframe=data_frame3.loc[:int(0.75*data_frame3.shape[0]),:]
test_dataframe=data_frame3.loc[int(0.75*data_frame3.shape[0]):,:]

yTrain=list(train_dataframe['class'])

yTest=list(test_dataframe['class'])
train_dataframe.drop(['class','subject'],axis=1,inplace=True)

test_dataframe.drop(['class','subject'],axis=1,inplace=True)
test_dataframe.reset_index(inplace=True,drop=True)

test_dataframe.head()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
train_title_column=list(train_dataframe['title'])

train_text_column=list(train_dataframe['text'])

test_title_column=list(test_dataframe['title'])

test_text_column=list(test_dataframe['text'])
train_dataframe.drop(['title','text'],axis=1,inplace=True)

test_dataframe.drop(['title','text'],axis=1,inplace=True)
count_vec=CountVectorizer(max_features=5000,ngram_range=(1,2))
train_title_sparse_matrix=count_vec.fit_transform(train_title_column)
test_title_sparse_matrix=count_vec.transform(test_title_column)
test_title_sparse_matrix.shape
train_dataframe_title = pd.DataFrame.sparse.from_spmatrix(train_title_sparse_matrix,columns=count_vec.get_feature_names())
test_dataframe_title=pd.DataFrame.sparse.from_spmatrix(test_title_sparse_matrix,columns=count_vec.get_feature_names())
train_dataframe_title.head()
test_dataframe_title.head()
train_dataframe1=pd.concat([train_dataframe,train_dataframe_title],axis='columns')
train_dataframe1.head()
test_dataframe1=pd.concat([test_dataframe,test_dataframe_title],axis='columns')
test_dataframe1.head()
count_vec_text=CountVectorizer(max_features=5000,ngram_range=(1,2))
train_text_sparse_matrix=count_vec_text.fit_transform(train_text_column)
test_text_sparse_matrix=count_vec_text.transform(test_text_column)
train_dataframe_text = pd.DataFrame.sparse.from_spmatrix(train_text_sparse_matrix,columns=count_vec_text.get_feature_names())
train_dataframe_text.head()
test_dataframe_text=pd.DataFrame.sparse.from_spmatrix(test_text_sparse_matrix,columns=count_vec_text.get_feature_names())
test_dataframe_text.head()
train_dataframe2=pd.concat([train_dataframe1,train_dataframe_text],axis='columns')
test_dataframe2=pd.concat([test_dataframe1,test_dataframe_text],axis='columns')
train_dataframe2.head()
test_dataframe2.head()
train_dataframe2.isnull().sum()
test_dataframe2.isnull().sum()
train_dataframe2.shape
test_dataframe2.shape
train_dataframe2.shape,test_dataframe2.shape,yTrain.shape,yTest.shape
xTrain=train_dataframe2.values

xTest=test_dataframe2.values
xTrain.shape,xTest.shape,yTrain.shape,yTest.shape
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=1000)

lr.fit(xTrain,yTrain)
yPredicted=lr.predict(xTest)
lr.score(xTest,yTest)
from sklearn.metrics import confusion_matrix,classification_report
print (confusion_matrix(yTest,yPredicted))
print (classification_report(yTest,yPredicted))
from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()

clf_rf.fit(xTrain,yTrain)
clf_rf.score(xTest,yTest)
yPredicted_rf=clf_rf.predict(xTest)
print (confusion_matrix(yTest,yPredicted_rf))
print (classification_report(yTest,yPredicted_rf))
from sklearn.naive_bayes import MultinomialNB
clf_mnb=MultinomialNB()
clf_mnb.fit(xTrain,yTrain)
clf_mnb.score(xTest,yTest)
yPredicted_mnb=clf_mnb.predict(xTest)
confusion_matrix(yTest,yPredicted_mnb)
print (classification_report(yTest,yPredicted_mnb))