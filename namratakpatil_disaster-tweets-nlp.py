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





#Importing the required libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

#Importing required libraries fo cleaning text
import re
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
#from sklearn.svm import SVC
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import VotingClassifier

print("Important libraries loaded successfully")
ds_train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
ds_test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print("Train and Test data sets are imported successfully")
#Checking number of records in train data set and few records from train data set
print("Number of records in Train data set",len(ds_train.index))
ds_train.head()
#Distinct keywords in train dataset
dist_keyword=ds_train['keyword'].value_counts()
dist_keyword
# creating bool series True for NaN values for location 
bool_series_location = pd.isnull(ds_train['location']) 

# filtering data  
# displaying data only with location = NaN  
ds_train[bool_series_location]
print("Number of records with missing location",len(ds_train[bool_series_location]))
# creating bool series True for NaN values  
bool_series_keyword = pd.isnull(ds_train['keyword']) 
# filtering data  
# displaying data only with Keywords = NaN  
ds_train[bool_series_keyword]
print("Number of records with missing keywords",len(ds_train[bool_series_keyword]))
# Calculate percentage of missing keywords
print('{}% of Kewords are missing from Total Number of Records'.format((len(ds_train[bool_series_keyword])/len(ds_train.index))*100))
#Check for Class Imbalance
target = ds_train['target']
sns.set_style('whitegrid')
sns.countplot(target)
# Plotting a bar graph of the number of tweets in each location, for the first ten locations listed
# in the column 'location'
location_count  = ds_train['location'].value_counts()
location_count = location_count[:10,]
plt.figure(figsize=(10,5))
sns.barplot(location_count.index, location_count.values, alpha=0.8)
plt.title('Top 10 locations')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('location', fontsize=12)
plt.show()
#original
data00=ds_train
#dropping unwanted column 'location'
ds_train=ds_train.drop(['location'],axis=1)
ds_train.head()
#dropping missing 'keyword' records from train data set
ds_train=ds_train.drop(ds_train[bool_series_keyword].index,axis=0)
#Resetting the index after droping the missing records
ds_train=ds_train.reset_index(drop=True)
print("Number of records after removing missing keywords",len(ds_train))
ds_train.head()
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

ds_train['text']=ds_train['text'].apply(lambda x : remove_html(x))
ds_train.head()
# Remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

ds_train['text']=ds_train['text'].apply(lambda x: remove_emoji(x))
ds_train['text']
#Remove Punctuation
import gensim
import string
def remove_punct(text):
    #print(text)
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

ds_train['text']=ds_train['text'].apply(lambda x : remove_punct(x))
ds_train['text']
import nltk
nltk.download('stopwords')
  

corpus  = []
pstem = PorterStemmer()
for i in range(ds_train['text'].shape[0]):
    #Remove unwanted words
    text = re.sub("[^a-zA-Z]", ' ', ds_train['text'][i])
    #Transform words to lowercase
    text = text.lower()
    text = text.split()
    #Remove stopwords then Stemming it
    text = [word for word in text if not word in set(stopwords.words('english')) ]
    text = ' '.join(text)
    #Append cleaned tweet to corpus
    corpus.append(text)
    
print("Corpus created successfully")  
#corpus
#Create dictionary 
uniqueWords = {}
for text in corpus:
    for word in text.split():
        if(word in uniqueWords.keys()):
            uniqueWords[word] += 1
        else:
            uniqueWords[word] = 1
            
#Convert dictionary to dataFrame
uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])
uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))
uniqueWords.head(10)
uniqueWords=uniqueWords[uniqueWords['WordFrequency']>=20]

#Create Bag of Words Model , here X represent bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = len(uniqueWords))
X = cv.fit_transform(corpus).todense()
y = ds_train['target'].values
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=2020)
print('Train Data splitted successfully')
# Fitting Gaussian Naive Bayes to the Training set
classifier_gnb = GaussianNB()
classifier_gnb.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_gnb = classifier_gnb.predict(X_test)
# Making the Confusion Matrix
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
#cm_gnb
sns.heatmap(cm_gnb,cmap="YlGnBu", annot=True,fmt='g')
#Calculating Model Accuracy
print('GaussianNB Classifier Accuracy Score is {} for Train Data Set'.format(classifier_gnb.score(X_train, y_train)))
print('GaussianNB Classifier Accuracy Score is {} for Test Data Set'.format(classifier_gnb.score(X_test, y_test)))
print('GaussianNB Classifier F1 Score is {}'.format(f1_score(y_test, y_pred_gnb)))
# Fitting XGBoost Model to the Training set
classifier_xgb = XGBClassifier(max_depth=6,learning_rate=0.3,n_estimators=1500,objective='binary:logistic',random_state=123,n_jobs=4)
classifier_xgb.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_xgb = classifier_xgb.predict(X_test)
# Making the Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
#print(cm_xgb)
sns.heatmap(cm_xgb,cmap="YlGnBu", annot=True,fmt="g")
print('XG Boost Model Accuracy Score for Train Data set is {}'.format(classifier_xgb.score(X_train, y_train)))
print('XG Boost Model Accuracy Score for Test Data set is {}'.format(classifier_xgb.score(X_test, y_test)))
print('XG Boost Model F1 Score is {}'.format(f1_score(y_test, y_pred_xgb)))


# Fitting multinomial naive bayes Model to the Training set
classifier_mnb = MultinomialNB()
classifier_mnb.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_mnb = classifier_mnb.predict(X_test)
# Making the Confusion Matrix
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
#print(cm_mnb)
sns.heatmap(cm_mnb,cmap="YlGnBu", annot=True,fmt="g")
print('MultinomialNB Model Accuracy Score for Train Data set is {}'.format(classifier_mnb.score(X_train, y_train)))
print('MultinomialNB Model Accuracy Score for Test Data set is {}'.format(classifier_mnb.score(X_test, y_test)))
print('MultinomialNB Model F1 Score is {}'.format(f1_score(y_test, y_pred_mnb)))
# Fitting Decision Tree Models to the Training set
classifier_dt = DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = None, 
                                           splitter='best', 
                                           random_state=55)
classifier_dt.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_dt = classifier_dt.predict(X_test)
# Making the Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
#print(cm_dt)
sns.heatmap(cm_dt,cmap="YlGnBu", annot=True,fmt="g")
#Calculating Model Accuracy
print('DecisionTree Model Accuracy Score for Train Data set is {}'.format(classifier_dt.score(X_train, y_train)))
print('DecisionTree Model Accuracy Score for Test Data set is {}'.format(classifier_dt.score(X_test, y_test)))
print('DecisionTree Model F1 Score is {}'.format(f1_score(y_test, y_pred_dt)))
# Fitting Gradient Boosting Models to the Training set
classifier_gb = GradientBoostingClassifier(loss = 'deviance',
                                                   learning_rate = 0.01,
                                                   n_estimators = 100,
                                                   max_depth = 30,
                                                   random_state=55)
classifier_gb.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_gb = classifier_gb.predict(X_test)
# Making the Confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)
#print(cm_gb)
sns.heatmap(cm_gb,cmap="YlGnBu", annot=True,fmt="g")
#Calculating Model Accuracy
print('Gradient Boosting Classifier Accuracy Score is {} for Train Data Set'.format(classifier_gb.score(X_train, y_train)))
print('Gradient Boosting Classifier Accuracy Score is {} for Test Data Set'.format(classifier_gb.score(X_test, y_test)))
print('Gradient Boosting Classifier F1 Score is {} '.format(f1_score(y_test, y_pred_gb)))
# Fitting Logistic Regression Model to the Training set
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)
# Predicting the Train data set results
y_pred_lr = classifier_lr.predict(X_test)
# Making the Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
#print(cm_lr)
sns.heatmap(cm_lr,cmap="YlGnBu", annot=True,fmt="g")
#Calculating Model Accuracy
print('Logistic Regression Model Accuracy Score for Train Data set is {}'.format(classifier_lr.score(X_train, y_train)))
print('Logistic Regression Model Accuracy Score for Test Data set is {}'.format(classifier_lr.score(X_test, y_test)))
print('Logistic Regression Model F1 Score is {}'.format(f1_score(y_test, y_pred_lr)))
#ds_test.isnull().sum()

#On Test Set#

ds_test=ds_test.drop(['location','keyword'],axis=1)
ds_test.head()



ds_test['text']=ds_test['text'].apply(lambda x : remove_html(x))


# Remove emojis


ds_test['text']=ds_test['text'].apply(lambda x: remove_emoji(x))

#Remove Punctuation


ds_test['text']=ds_test['text'].apply(lambda x : remove_punct(x))



corpus  = []
pstem = PorterStemmer()
for i in range(ds_test['text'].shape[0]):
    #Remove unwanted words
    text = re.sub("[^a-zA-Z]", ' ', ds_test['text'][i])
    #Transform words to lowercase
    text = text.lower()
    text = text.split()
    #Remove stopwords then Stemming it
    text = [word for word in text if not word in set(stopwords.words('english')) ]
    text = ' '.join(text)
    #Append cleaned tweet to corpus
    corpus.append(text)
    
print("Corpus created successfully")  


#Create dictionary 
uniqueWords = {}
for text in corpus:
    for word in text.split():
        if(word in uniqueWords.keys()):
            uniqueWords[word] += 1
        else:
            uniqueWords[word] = 1
            
#Convert dictionary to dataFrame
uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])
uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)
print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))
uniqueWords.head(10)
uniqueWords=uniqueWords[uniqueWords['WordFrequency']>=20]

#Create Bag of Words Model , here X represent bag of words

X_test = cv.fit_transform(corpus).todense()


X_test.shape
X_train.shape
y_pred = classifier_lr.predict(X_test)

model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
model_submission['target'] = np.round(y_pred).astype('int')
model_submission.to_csv('model_submission1.csv', index=False)
model_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
#model_submission.head()
model_submission['target']
#model_submission.shape
#for downloading submission file #

from subprocess import check_output
print(check_output(["ls", "../working"]).decode("utf8"))

from IPython.display import FileLink
FileLink('model_submission1.csv')
