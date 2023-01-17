import pandas as pd

import sqlite3

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import LocalOutlierFactor

from nltk.corpus import stopwords

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



# Importing the data

df = pd.read_csv('../input/Reviews.csv')

data = df[df.Score!=3][0:5000]



# Performing Basic Checks

print('The Shape of the data'+str(data.shape)+'\n')

print('Data Types of all columns\n',data.dtypes,'\n')

print('Checking for Na Values\n',data.isna().sum())

data.head(5)
# Converting score to 1(positive) or 0(negative)

def score(x):

    '''This function converts the score into positive or negative'''

    if x<3:

        return 0

    elif x>3:

        return 1

positivenegative = data['Score'].map(score)

data['score'] = positivenegative

data.drop('Score',axis=1,inplace=True)

data.head(5)    



# Checkes: 1. Numerator<=Denominator

print(data[data.HelpfulnessNumerator>data.HelpfulnessDenominator])



# 2. Duplication

data[data.duplicated(subset={'ProfileName', 'HelpfulnessNumerator',

       'HelpfulnessDenominator', 'Time', 'Summary', 'Text', 'score'})]

data[data.Text==data['Text'].iloc[2309]]
# Deleting the values

data.drop_duplicates(keep='first',subset={'ProfileName', 'HelpfulnessNumerator',

       'HelpfulnessDenominator', 'Time', 'Summary', 'Text', 'score'},inplace=True)

print(data[data.duplicated(subset={'ProfileName','HelpfulnessNumerator',

       'HelpfulnessDenominator', 'Time', 'Summary', 'Text', 'score'})])

print(data.shape)

# All the duplicate values have been removed
# Distribution of scores

print(data.score.value_counts())

c=[]

for i in data.score.value_counts():c.append(i)

plt.bar(['1','0'],c)

plt.show()

from nltk.stem import SnowballStemmer

import nltk.stem

import re

import string



stop = set(stopwords.words('english'))

sno = nltk.SnowballStemmer('english')



 # Defining functions that will make our cleaning easier

def cleanhtml(sent):

    '''This function cleans the html tags ina  sentence'''

    cleanr = re.compile('<.*?>')

    clean_sentence = re.sub(cleanr,'',sent)

    return clean_sentence



def cleanpunc(word):

    '''This function cleans the punctuations in a word'''

    clean_word = re.sub(r'[?|!|,|.|\|/|\'|"|:|;|#]',r'',word)

    return clean_word
cleaned_text =[] # List of the cleaned reviews

for sentence in data['Text'].values:

    clean_sentence = []

    sentence = cleanhtml(sentence)

    for word in sentence.split():

        cleaned_word = cleanpunc(word)

        if (cleaned_word.lower() not in stop) & (cleaned_word.isalpha()):

            if len(cleaned_word)>2:

                stemmed_word = sno.stem(cleaned_word.lower()).encode('utf8')

                clean_sentence.append(stemmed_word)

            



    str1=b' '.join(clean_sentence)

    #print(str1)

    cleaned_text.append(str1)

print('Cleaned Text: ',cleaned_text[0])

print('\nActual Text: ',data['Text'].values[0])

data['cleanedText'] = cleaned_text

data.head(5)
from sklearn.model_selection import train_test_split

y = data['score']

data.drop('score',axis=1,inplace = True)

X = data

print(X.shape)

print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,shuffle =False) # 33% data in test

X_train,X_cv,y_train,y_cv = train_test_split(X_train,y_train,test_size = 0.33,shuffle =False) # 33% data in cross validation

print('Train shape ',X_train.shape,y_train.shape)

print('CV shape',X_cv.shape,y_cv.shape)

print('Test shape',X_test.shape,y_cv.shape)

# Converting cleaned text into bow vectors



bow = CountVectorizer(max_features = 500) #ngram = 1





Xtrain_bow = bow.fit_transform(X_train['cleanedText'])

print(bow.get_feature_names()[0:10])

#print(bow.vocabulary_)



Xcross_bow = bow.transform(X_cv['cleanedText'])

Xtest_bow = bow.transform(X_test['cleanedText'])



# Checking their dimension

print(Xtrain_bow.shape)

print(Xcross_bow.shape)

print(Xtest_bow.shape)

# Reduced the reviews bow vectors with 500 features



# Standardizing

from sklearn.preprocessing import StandardScaler

std = StandardScaler()



Xtrain_bow_std = std.fit_transform(Xtrain_bow.toarray()) # converting sparse matrix into dense matrix using toarray()

Xtest_bow_std = std.fit_transform(Xtest_bow.toarray())

Xcross_bow_std = std.fit_transform(Xcross_bow.toarray())

# cheking if train and test have same distribution ~ Not taking cross validation as if train and test have same distribution we can assume CV will also have similar distribution.



# 1. Creating the new dataset train labels = 1, test labels = 0

Train_label = np.ones(len(X_train))

Test_label = np.zeros(len(X_test))

labels = np.hstack((Train_label.T,Test_label.T))

#Train

new_data_train = (np.vstack((Xtrain_bow_std.T,y_train)).T)

print('Dimensions of train dataset incusing original labels',new_data_train.shape)

#print(new_data_train[0:10])



# Test

new_data_test = (np.vstack((Xtest_bow_std.T,y_test)).T)

print('\nDimensions of train dataset incusing original labels',new_data_test.shape)

#print(new_data_test[0:10])



# 2. Combine the train and test data

dist_data = np.vstack((new_data_train,new_data_test))

print('\nThe shape of combined new data',dist_data.shape)



# 3. Random splitting into train and test for modeling

x_train,x_test,Y_train,Y_test = train_test_split(dist_data,labels,test_size=0.33,shuffle=True)

print('\nDimension of train data with label=1',x_train.shape,Y_train.shape)

print('\nDimension of test data with label=0',x_test.shape,Y_test.shape)



# 4. Modelling using KNN

from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors = 20)

knn.fit(x_train,Y_train)

test_predict = knn.predict(x_test)



#5. Inspect Accuracy

accuracy = accuracy_score(test_predict,Y_test)

print('\nAccuracy of mdoel',accuracy)
from imblearn.over_sampling import SMOTE

print('Number of positive and negative reviews:\n',y_train.value_counts())

sm = SMOTE(random_state=0,ratio=1.0)

Xtrain_res,ytrain_res = sm.fit_sample(Xtrain_bow_std,y_train)

print(np.bincount(ytrain_res),Xtrain_res.shape) # equal 1s and 0s
import scipy

from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier



neighbors = [1,2,5,10,20,25,30,40,50]

auc_scores_cv = []

auc_scores_train = []

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(Xtrain_res,ytrain_res) # fitting the model

    

    cv_predict = knn.predict_proba(Xcross_bow_std) # predicting the probabilistic values cross validation set

    cv_auc = roc_auc_score(y_cv,cv_predict[:,1])

    auc_scores_cv.append(cv_auc) #auc value for CV set

    

    train_predict = knn.predict_proba(Xtrain_bow_std) # predicting on train itself

    train_auc = roc_auc_score(y_train,train_predict[:,1]) # auc value for train

    auc_scores_train.append(train_auc)



print('Train AUC Scores ',auc_scores_train)

print('CV AUC Scores',auc_scores_cv)

error_cv = [1-  x for x in auc_scores_cv]

error_train = [1 - x for x in auc_scores_train]

# Visualising Train error and Cross Validation Error

plt.figure(figsize=(7,7))

plt.plot(neighbors,error_cv,color = 'r',label='Cross-Validation error') 

plt.plot(neighbors,error_train,color='b',label='Train Error')

plt.xlabel('K Neighbors')

plt.ylabel('error')

plt.title('K: Hyperparameter')

plt.legend(loc='lower right')

plt.grid()

plt.show()



# The optimal value of K

best_k =10
from sklearn.metrics import roc_curve,auc

from sklearn.metrics import confusion_matrix

# Final Prediction using Test Data

knn = KNeighborsClassifier(n_neighbors = best_k)

knn.fit(Xtrain_res,ytrain_res) # fitting the model in train set



# predicted values determination

predicted_values_train = knn.predict_proba(Xtrain_bow_std)

predicted_values_test = knn.predict_proba(Xtest_bow_std)

predicted_values_cv = knn.predict_proba(Xcross_bow_std)



# False Positive Rate and True Positive Rate

train_fpr,train_tpr,thresholds = roc_curve(y_train,predicted_values_train[:,1],pos_label=1)

cv_fpr,cv_tpr,thresholds = roc_curve(y_cv,predicted_values_cv[:,1],pos_label=1)

test_fpr,test_tpr,thresholds = roc_curve(y_test,predicted_values_test[:,1],pos_label=1)



# Visualising ROC

plt.figure(figsize=(7,7))

plt.plot(train_fpr,train_tpr,color='g',label='Train AUC = '+str(auc(train_fpr,train_tpr)))

plt.plot(cv_fpr,cv_tpr,color='b',label = 'CV AUC = '+str(auc(cv_fpr,cv_tpr)))

plt.plot(test_fpr,test_tpr,color='r',label = 'Test AUC = '+str(auc(test_fpr,test_tpr)))

plt.legend(loc = 'lower right')

plt.show()



# Confucion Matrix



print('Train:\n',confusion_matrix(y_train,knn.predict(Xtrain_bow_std)).T)

print('\nCV:\n',confusion_matrix(y_cv,knn.predict(Xcross_bow_std)).T)

print('\nTest:\n',confusion_matrix(y_test,knn.predict(Xtest_bow_std)).T)

cm = confusion_matrix(y_test,knn.predict(Xtest_bow_std)).T

print('#'*50)

print('TNR for Test = ',(cm[0][0])/(cm[0][0] + cm[1][0]) )

print('FPR for Test = ',cm[1][0]/(cm[1][0]+cm[0][0]) )
