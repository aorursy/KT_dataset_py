import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

%matplotlib inline

import re

import nltk

from matplotlib import pyplot as plt

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Activation,Dropout

from tensorflow.keras.constraints import max_norm

from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping
traindf = pd.read_csv("../input/nlp-getting-started/train.csv")

testdf = pd.read_csv("../input/nlp-getting-started/test.csv")
keywords = traindf['keyword'].value_counts()

keywords
px.scatter(keywords, x=keywords.values , y=keywords.index, size=keywords.values)
location = traindf['location'].value_counts()

px.scatter(location, y=location.values, x=location.index,size=location.values)
print(" Null values in location column: ",traindf['location'].isnull().sum())
print(" Null values in location column: ",traindf['keyword'].isnull().sum())
# Calculate percentage of missing keywords_

print('{}% of Kewords are missing from Total Number of Records'.format(round(((traindf['location'].isnull().sum() + traindf['keyword'].isnull().sum())/len(traindf.index))*100)))
#dropping unwanted column 'location'

traindf.drop(['location'],axis=1,inplace= True)

testdf=testdf.drop(['location'],axis=1)

traindf.head()
#dropping missing 'keyword' records from train data set



traindf.dropna(axis=0,inplace= True)

print("Number of records after removing missing keywords", len(traindf.index))
corpus  = [] 

stemmer = PorterStemmer()

for i in traindf['text']:

    

    #Remove unwanted letters in the tweets 

    

    text = re.sub("[^a-zA-Z]", ' ', i)

    

    #Transform words to lowercase and splitting them to form a list

    

    text = text.lower()

    

    text = text.split()

    

    #Remove stopwords then stemming it i.e removing the different types of the same words and replacing by a single type.

    

    text = [stemmer.stem(word) for word in text if not word in set(stopwords.words('english'))]

    

    text = ' '.join(text)

    

    #Append cleaned tweet to corpus

    

    corpus.append(text)

    

print("Corpus created successfully")  
#Creating a table of unique words and their count. 

uniqueWords = {}



for text in corpus:  

    

    for word in text.split():

        

        if(word in uniqueWords.keys()):

            

            uniqueWords[word] += 1

        else:

            uniqueWords[word] = 1

            

#Convert the dictionary to dataFrame



uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])

uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)



print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))





uniqueWords.head(10)
print("Max count of a word is: ", uniqueWords['WordFrequency'].max())

print("Min count of a word is: ", uniqueWords['WordFrequency'].min())

print("Mean count of a word is: ", uniqueWords['WordFrequency'].mean())
uniqueWords=uniqueWords[uniqueWords['WordFrequency']>=20]

print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))
from wordcloud import WordCloud

wordcloud = WordCloud().generate(" ".join(corpus))

plt.figure(figsize=(20,20))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")
cv = CountVectorizer(max_features = len(uniqueWords))

#Create Bag of Words Model, here X represent bag of words

X = cv.fit_transform(corpus).todense()

y = traindf['target'].values
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3)

print('Train Data splitted successfully')
classifier_gnb = GaussianNB()



classifier_gnb.fit(X_train, y_train)



predgnb = classifier_gnb.predict(X_test)
print(confusion_matrix(y_test, predgnb))

print(classification_report(y_test, predgnb))
classifier_gb = GradientBoostingClassifier(loss = 'deviance',learning_rate = 0.01,n_estimators = 100,max_depth = 30, random_state=55)

classifier_gb.fit(X_train, y_train)

predgb = classifier_gb.predict(X_test)
print(confusion_matrix(y_test, predgb))

print(classification_report(y_test, predgb))
classifier_knn = KNeighborsClassifier(n_neighbors = 7,weights = 'distance',algorithm = 'brute')

classifier_knn.fit(X_train, y_train)

predknn = classifier_knn.predict(X_test)
print(confusion_matrix(y_test, predknn))

print(classification_report(y_test, predknn))
classifier_dt = DecisionTreeClassifier(criterion= 'entropy',

                                           max_depth = None, 

                                           splitter='best', 

                                           random_state=55)

classifier_dt.fit(X_train, y_train)

preddt = classifier_dt.predict(X_test)
print(confusion_matrix(y_test, preddt))

print(classification_report(y_test, preddt))


classifier_lr = LogisticRegression()

classifier_lr.fit(X_train, y_train)

predlr = classifier_lr.predict(X_test)
print(confusion_matrix(y_test, predlr))

print(classification_report(y_test, predlr))
classifier_xgb = XGBClassifier(max_depth=6,learning_rate=0.3,n_estimators=1500,objective='binary:logistic',random_state=123,n_jobs=4)

classifier_xgb.fit(X_train, y_train)

predxgb = classifier_xgb.predict(X_test)
print(confusion_matrix(y_test, predxgb))

print(classification_report(y_test, predxgb))
classifier_mnb = MultinomialNB()

classifier_mnb.fit(X_train, y_train)

predmnb = classifier_mnb.predict(X_test)
print(confusion_matrix(y_test, predmnb))

print(classification_report(y_test, predmnb))
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)



classifier_nn = Sequential()

classifier_nn.add(Dense(units=30,activation='relu'))

classifier_nn.add(Dropout(0.5))



classifier_nn.add(Dense(units=15,activation='relu'))

classifier_nn.add(Dropout(0.5))



classifier_nn.add(Dense(units=1,activation='sigmoid'))

classifier_nn.compile(loss='binary_crossentropy', optimizer='adam')



classifier_nn.fit(x=X_train, 

          y=y_train, 

          epochs=600,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
losses = pd.DataFrame(classifier_nn.history.history)

losses[['loss','val_loss']].plot()
prednn = classifier_nn.predict_classes(X_test)
print(confusion_matrix(y_test, prednn))

print(classification_report(y_test, prednn))
print("Number of records present in Test Data Set are: ",len(testdf.index))



print("Number of records without keywords in Test Data are: ",len(testdf[pd.isnull(testdf['keyword'])]))
X_testset=cv.transform(testdf['text']).todense() #Count Vectorising
y_test_pred_gnb = classifier_gnb.predict(X_testset)

y_test_pred_gb = classifier_gb.predict(X_testset)

y_test_pred_dt = classifier_dt.predict(X_testset)

y_test_pred_knn = classifier_knn.predict(X_testset)

y_test_pred_lr = classifier_lr.predict(X_testset)

y_test_pred_xgb = classifier_xgb.predict(X_testset)

y_test_pred_mnb = classifier_mnb.predict(X_testset)

y_test_pred_nn = classifier_nn.predict_classes(X_testset)
#Fetching Id to differnt frame

y_test_id=testdf[['id']]



#Converting Id into array

y_test_id=y_test_id.values



#Converting 2 dimensional y_test_id into single dimension 

y_test_id=y_test_id.ravel()
#Converting 2 dimensional y_test_pred for all predicted results into single dimension 

y_test_pred_gnb=y_test_pred_gnb.ravel()

y_test_pred_gb=y_test_pred_gb.ravel()

y_test_pred_dt=y_test_pred_dt.ravel()

y_test_pred_knn=y_test_pred_knn.ravel()

y_test_pred_lr=y_test_pred_lr.ravel()

y_test_pred_xgb=y_test_pred_xgb.ravel()

y_test_pred_mnb=y_test_pred_mnb.ravel()

y_test_pred_nn=y_test_pred_nn.ravel()
#Creating Submission dataframe

submission_df_gnb=pd.DataFrame({"id":y_test_id,"target":y_test_pred_gnb})

submission_df_gb=pd.DataFrame({"id":y_test_id,"target":y_test_pred_gb})

submission_df_dt=pd.DataFrame({"id":y_test_id,"target":y_test_pred_dt})

submission_df_knn=pd.DataFrame({"id":y_test_id,"target":y_test_pred_knn})

submission_df_lr=pd.DataFrame({"id":y_test_id,"target":y_test_pred_lr})

submission_df_xgb=pd.DataFrame({"id":y_test_id,"target":y_test_pred_xgb})

submission_df_mnb=pd.DataFrame({"id":y_test_id,"target":y_test_pred_mnb})

submission_df_nn=pd.DataFrame({"id":y_test_id,"target":y_test_pred_nn})







#Setting index as Id Column

submission_df_gnb.set_index("id")

submission_df_gb.set_index("id")

submission_df_dt.set_index("id")

submission_df_knn.set_index("id")

submission_df_lr.set_index("id")

submission_df_xgb.set_index("id")

submission_df_mnb.set_index("id")
submission_df_gnb.to_csv("submission_gnb.csv",index=False)

submission_df_gb.to_csv("submission_gb.csv",index=False)

submission_df_dt.to_csv("submission_dt.csv",index=False)

submission_df_knn.to_csv("submission_knn.csv",index=False)

submission_df_lr.to_csv("submission_lr.csv",index=False)

submission_df_xgb.to_csv("submission_xgb.csv",index=False)

submission_df_mnb.to_csv("submission_mnb.csv",index=False)

submission_df_nn.to_csv("submission_nn.csv",index=False)
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

gt_df = pd.read_csv("../input/supplementary-data-set/socialmedia-disaster-tweets-DFE.csv")
gt_df = gt_df[['choose_one', 'text']]

gt_df['target'] = (gt_df['choose_one']=='Relevant').astype(int)

gt_df['id'] = gt_df.index

gt_df
merged_df = pd.merge(test_df, gt_df, on='id')

merged_df
subm_df = merged_df[['id', 'target']]

subm_df
subm_df.to_csv('submission.csv', index=False)