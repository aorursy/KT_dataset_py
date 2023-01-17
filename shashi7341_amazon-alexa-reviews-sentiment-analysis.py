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
#important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
#Data Acqisition
data=pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv', sep='\t')
data.shape
#head
data.head()
#information
data.info()
#data description with statistic measures
data.describe()
#Checking for count of null values of each feature
data.isnull().sum()
#Number of unique values of each column
data.nunique().plot(kind='bar', figsize=(10,6), color='brown')
plt.xlabel('Features')
plt.ylabel('Count')
plt.title('Number of Unique Values of each Feature ')
plt.figure(figsize=(10,6))
sns.countplot(data['rating'],palette='Set3')
data['date'].value_counts()[:20].plot(kind='bar',figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Count')
plt.figure(figsize=(14,8))
barplot=sns.countplot(data['variation'], palette='Set2')
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
plt.figure(figsize=(12,6))
sns.countplot(data['feedback'], palette='Set1')
plt.figure(figsize=(12,8))
sns.countplot(data['rating'], hue=data['feedback'], palette='Set2')
plt.figure(figsize=(14,8))
barplot=sns.countplot(data['variation'],hue=data['feedback'], palette='Set2')
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
#Length of the string of verified reviews
df=data
df['length']=df['verified_reviews'].apply(len)
df.head()
#box plot on length of reviews
plt.figure(figsize=(12,6))
sns.boxplot(df['length'], palette='Set3')
maximum_length=max(df['length'])
print('Maximum length Review from overall verified reviews is: ',maximum_length)
df[df['length']==2851]['verified_reviews'].iloc[0]
df[df['length']==2851]['feedback'].iloc[0]
positive_feedbacks=data[data['feedback']==1]['verified_reviews']
negative_feedbacks=data[data['feedback']==0]['verified_reviews']
#Word Cloud for positive feedback reviews
from wordcloud import WordCloud
plt.figure(figsize=(14,8))
wordcloud1=WordCloud(width=400,height=300, contour_color='black').generate(' '.join(positive_feedbacks))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title('Positive Feedback Reviews',fontsize=40)
#Word Cloud for Negative feedback reviews
plt.figure(figsize=(14,8))
wordcloud2=WordCloud(width=400,height=300, contour_color='black').generate(' '.join(negative_feedbacks))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title('Negative Feedback Reviews',fontsize=40)
#Removing leading and ending spaces of string
data['verified_reviews']=data['verified_reviews'].apply(lambda x: x.strip())
#Text Lowercasing
data['verified_reviews']=data['verified_reviews'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
data['verified_reviews'].head()
#removing punctuations from string
import string
punct_dict=dict((ord(punct),None) for punct in string.punctuation)
print(string.punctuation)
print(punct_dict)
for i in range(0, data.shape[0]):
    data['verified_reviews'][i]=data['verified_reviews'][i].translate(punct_dict)
data['verified_reviews'].head()    
#Removing Stop Words
from nltk.corpus import stopwords
stop=stopwords.words('english')
print(len(stop))
data['verified_reviews']=data['verified_reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['verified_reviews'].head()
#Removing Numbers
data['verified_reviews']=data['verified_reviews'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
data['verified_reviews'].head()
#Lemmatization
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
data['verified_reviews']=data['verified_reviews'].apply(lambda x: ''.join(lem.lemmatize(term) for term in x))
data['verified_reviews'].head()
#Removing URL's
data['verified_reviews'] = data['verified_reviews'].str.replace('http\S+|www.\S+','',case=False)
data['verified_reviews'].head()
#List of Tokens-Tokenization
from nltk.tokenize import word_tokenize
all_words=[]
for msg in data['verified_reviews']:
    words=word_tokenize(msg)
    for w in words:
        all_words.append(w)  
#Frequency of Most Common Words
import nltk
frequency_dist=nltk.FreqDist(all_words)
print('Length of the words',len(frequency_dist))
print('Most Common Words',frequency_dist.most_common(100))
#Frequency Plot for first 100 most frequently occuring words
plt.figure(figsize=(20,8))
frequency_dist.plot(100,cumulative=False)
#### Feature extraction using Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVect=TfidfVectorizer(max_features=1500,stop_words='english')

X=tfidfVect.fit_transform(data['verified_reviews']).toarray()
y=data.iloc[:,4].values
print(tfidfVect.get_feature_names())
#Shape of tfidf vectorizer
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)
from sklearn.naive_bayes import MultinomialNB
model1=MultinomialNB()
model1.fit(X_train, y_train)
y_pred=model1.predict(X_test)
#Important Metrics to know the Performance of Model
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print('Classification Report',classification_report(y_test,y_pred))
print('Confusion Matrix',confusion_matrix(y_test,y_pred))
print('Accuracy Score',accuracy_score(y_test,y_pred))
print('Precision Score',precision_score(y_test,y_pred))
print('Recall Score',recall_score(y_test,y_pred))
print('F1 Score',f1_score(y_test,y_pred))
#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(model1,X=X_train,y=y_train,cv=5)
from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
#Important Metrics to know the Performance of Model
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print('Classification Report',classification_report(y_test,y_pred2))
print('Confusion Matrix',confusion_matrix(y_test,y_pred2))
print('Accuracy Score',accuracy_score(y_test,y_pred2))
print('Precision Score',precision_score(y_test,y_pred2))
print('Recall Score',recall_score(y_test,y_pred2))
print('F1 Score',f1_score(y_test,y_pred2))
#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(model2,X=X_train,y=y_train,cv=5)
from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier()
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)
#Important Metrics to know the Performance of Model
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
print('Classification Report',classification_report(y_test,y_pred3))
print('Confusion Matrix',confusion_matrix(y_test,y_pred3))
print('Accuracy Score',accuracy_score(y_test,y_pred3))
print('Precision Score',precision_score(y_test,y_pred3))
print('Recall Score',recall_score(y_test,y_pred3))
print('F1 Score',f1_score(y_test,y_pred3))
#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(model3,X=X_train,y=y_train,cv=5)
