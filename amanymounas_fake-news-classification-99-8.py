import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('darkgrid')

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords    
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

from keras.preprocessing import text,sequence
import re,string,unicodedata
real = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
real.head()
fake.head()
#create a target column of 0 or 1 if the new is fake or real 
real['Label']=0
fake['Label']=1

#Using conactenate function of pandas :
data=pd.concat([real,fake])
data.sample(5)


#check the shape of data
np.shape(data)
#check for null values --> no null values found!
data.isnull().sum()
#How many of the given news are fake and how many of them are real?
sns.countplot(data.Label)
#how many unqiue titles are there. Are any of the titles repeated?
data.title.count()
data.subject.value_counts()
plt.figure(figsize=(10,10))
chart=sns.countplot(x='subject',hue='Label',data=data,palette='muted')
chart.set_xticklabels(chart.get_xticklabels(),rotation=90,fontsize=10)

X = data['text']+ " " + data['title']
Y = data['Label']
print(X)
print(Y)
stop_words=set(stopwords.words('english'))
punctuation=list(string.punctuation)
stop_words.update(punctuation)

def clean_text_data(text):
    text = BeautifulSoup(text,"html.parser").get_text()
    text = re.sub('\[[^]]*\]','',text)
    text = re.sub(r'http\S+','',text)
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())
    return " ".join(final_text)  


X = X.apply(clean_text_data)
print(X)
#split the train and test data
X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(max_features=5000,ngram_range=(1,3))


tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)



from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB(alpha=0.1)

classifier.fit(tfidf_train, y_train)
pred = classifier.predict(tfidf_test)

print(f'Accuracy: {round((accuracy_score(y_test, pred))*100,2)}%')

cm = confusion_matrix(y_test, pred)

cm=pd.DataFrame(cm,index=['Fake','Not Fake'],columns=['Fake','Not Fake'])


plt.figure(figsize=(5,5))
sns.heatmap(cm,cmap="Blues",linecolor='black',linewidth=1,annot=True,fmt='',xticklabels=['Fake','Not Fake'],yticklabels=['Fake','Not Fake'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

clf = PassiveAggressiveClassifier(max_iter=80)

clf.fit(tfidf_train,y_train)

y_pred=clf.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

cm=confusion_matrix(y_test,y_pred)
cm=pd.DataFrame(cm,index=['Fake','Not Fake'],columns=['Fake','Not Fake'])


plt.figure(figsize=(5,5))
sns.heatmap(cm,cmap="Blues",linecolor='black',linewidth=1,annot=True,fmt='',xticklabels=['Fake','Not Fake'],yticklabels=['Fake','Not Fake'])
plt.xlabel('Actual')
plt.ylabel('Predicted')