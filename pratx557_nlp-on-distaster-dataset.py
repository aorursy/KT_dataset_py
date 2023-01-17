import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
df=pd.read_csv("../input/traincsv/train.csv") #Importing Data
df.head()
df2=df.drop(['keyword','location','id'],axis=1)
text=df['text'].iloc[0]
from bs4 import BeautifulSoup
df1=BeautifulSoup(text,'html.parser')
soup=df1.get_text()
soup
import re

text = re.sub('\[[^]]*\]', ' ',text )
text = re.sub('[^a-zA-Z]', ' ', text)
text
text.lower()
stopw=stopwords.words('english')
stopw
text=text.split()
text
text = [word for word in text if not word in set(stopw)]
text
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
review_s = [ps.stem(word) for word in text]
review_s
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
text = [lem.lemmatize(word) for word in text]
text
text = ' '.join(text)
text
corpus=[]
corpus.append(text)
count_vec=CountVectorizer()
text_count_vec=count_vec.fit_transform(corpus)
text_count_vec.toarray()
count_vec_bin = CountVectorizer(binary=True)
text_count_vec_bin = count_vec_bin.fit_transform(corpus)

text_count_vec_bin.toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer()
text_tfidf_vec=tfidf_vec.fit_transform(corpus)
text_tfidf_vec.toarray()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
dataset_train,dataset_test,train_data_label,test_data_label=train_test_split(df2['text'],df2['target'],test_size=.30,random_state=42)
train_data_label
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

corpus_train=[]
corpus_test=[]
for i in range(dataset_train.shape[0]):
    soup = BeautifulSoup(dataset_train.iloc[i], "html.parser")
    text = soup.get_text()
    text = re.sub('\[[^]]*\]', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text]
    text = ' '.join(text)
    corpus_train.append(text)
    
for j in range(dataset_test.shape[0]):
    soup = BeautifulSoup(dataset_test.iloc[j], "html.parser")
    text = soup.get_text()
    text = re.sub('\[[^]]*\]', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text]
    text = ' '.join(text)
    corpus_test.append(text)
tfidf_vec = TfidfVectorizer(ngram_range=(1, 3))

tfidf_vec_train = tfidf_vec.fit_transform(corpus_train)
tfidf_vec_test = tfidf_vec.transform(corpus_test)
from sklearn.svm import LinearSVC
Linear_svc=LinearSVC(C=.5,random_state=42)
Linear_svc.fit(tfidf_vec_train,train_data_label)
predict = Linear_svc.predict(tfidf_vec_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Classification Report: \n", classification_report(test_data_label, predict))
print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict))
print("Accuracy: \n", accuracy_score(test_data_label, predict))
dataset_predict = dataset_test.copy()
dataset_predict = pd.DataFrame(dataset_predict)
dataset_predict.columns = ['Message']
dataset_predict = dataset_predict.reset_index()
dataset_predict = dataset_predict.drop(['index'], axis=1)
dataset_predict.head()
test_actual_label = test_data_label.copy()
test_actual_label = pd.DataFrame(test_actual_label)
test_actual_label.columns = ['id']
test_actual_label['id'] = test_actual_label['id']
test_predicted_label = predict.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment']
test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
test_result.head(30)
