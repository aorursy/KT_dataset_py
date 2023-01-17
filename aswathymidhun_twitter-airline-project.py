import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

data_file = '../input/twitter-airline-sentiment/Tweets.csv'
data = pd.read_csv(data_file)

data.head()
data.tail()
data.shape
data.info()
data.columns
data.isnull()
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.airline.value_counts().plot(kind='pie',autopct='%1.0f')
data.airline_sentiment.value_counts().plot(kind='pie',autopct='%1.0f')
sns.countplot(x='airline_sentiment',data=data,palette='viridis')
plt.figure(figsize=(12,7))
sns.countplot(x='airline',hue='airline_sentiment',data=data,palette='rainbow')
sns.barplot(x='airline_sentiment',y='airline_sentiment_confidence',data=data,palette='viridis')
sns.boxplot(x='airline',y='airline_sentiment_confidence',data=data)
Features = data.iloc[:,10].values
Labels = data.iloc[:,1].values
processed_Features = []

for sentence in range(0,len(Features)):
    #remove all the special character
    processed_Feature = re.sub(r'\W',' ',str(Features[sentence]))

    #remove all single characters
    processed_Feature = re.sub(r'\s+[a-zA-Z]\s+',' ',processed_Feature)

    #remove single characters from the start
    processed_Feature = re.sub(r'\^[a-zA-Z]\s+',' ',processed_Feature)

    #substituting multiple spaces with single space
    processed_Feature = re.sub(r'\s+',' ',processed_Feature,flags=re.I)

    #Removing prefixed 'b'
    processed_Feature = re.sub(r'^b\s+', ' ',processed_Feature)

    #converrting into lowercase
    processed_Feature = processed_Feature.lower()
    processed_Features.append(processed_Feature)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
!pip install --user -U nltk
import nltk
nltk.download('stopwords')
vectorizer = TfidfVectorizer(max_features=2500,min_df=7,max_df=0.8,stop_words=stopwords.words('english'))
processed_Features = vectorizer.fit_transform(processed_Features).toarray()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(processed_Features,Labels,test_size=0.2,random_state=0)
import sys
print(sys.version)
from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200,random_state=0)
text_classifier.fit(x_train,y_train)
predictions = text_classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,predictions))
print('accuracy score',accuracy_score(y_test,predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.svm import SVC
clf = SVC(kernel='linear',random_state=1)
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,predictions))
print('accuracy score',accuracy_score(y_test,predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(y_test,predictions))

print('accuracy score',accuracy_score(y_test,predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

