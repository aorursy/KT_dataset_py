import pandas as pd
import numpy as np
df=pd.read_csv('../input/pratik/prat.csv',error_bad_lines=False)
df=df[['Message','Category']]
df.head(20)
df.head()
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
Message=df['Message'].iloc[0]
Message
from bs4 import BeautifulSoup
df1=BeautifulSoup(Message,'html.parser')
soup=df1.get_text()
soup
import re
message=re.sub('\[[^]]*\']', ' ',Message)
message=re.sub('[^a-zA-Z]', ' ',Message)
message
message.lower()
stopw=stopwords.words('english')
message=message.split()
message
message=[word for word in message if not word in set(stopw)]
message
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
message_s=[ps.stem(word) for word in message]
message
from nltk.stem import WordNetLemmatizer

lem=WordNetLemmatizer()
message=[lem.lemmatize(word) for word in message]
message
message = ' '.join(message)
message
corpus=[]
corpus.append(message)
count_vec=CountVectorizer()
message_count_vec=count_vec.fit_transform(corpus)
message_count_vec.toarray()
count_vec_bin = CountVectorizer(binary=True)
message_count_vec_bin = count_vec_bin.fit_transform(corpus)

message_count_vec_bin.toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer()
message_tfidf_vec=tfidf_vec.fit_transform(corpus)
message_tfidf_vec.toarray()

from sklearn.model_selection import train_test_split
dataset_train,dataset_test,train_data_label,test_data_label=train_test_split(df['Message'],df['Category'],test_size=.30,random_state=42)
train_data_label = (train_data_label.replace({'spam': 1, 'ham': 0})).values
test_data_label  = (test_data_label.replace({'spam': 1, 'ham': 0})).values
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

corpus_train=[]
corpus_test=[]

for i in range(dataset_train.shape[0]):
    soup = BeautifulSoup(dataset_train.iloc[i], "html.parser")
    message = soup.get_text()
    message = re.sub('\[[^]]*\]', ' ', Message)
    message = re.sub('[^a-zA-Z]', ' ', Message)
    message = message.lower()
    message = message.split()
    message = [word for word in message if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    message = [lem.lemmatize(word) for word in message]
    message = ' '.join(message)
    corpus_train.append(message)
    
for j in range(dataset_test.shape[0]):
    soup = BeautifulSoup(dataset_test.iloc[j], "html.parser")
    message = soup.get_text()
    message = re.sub('\[[^]]*\]', ' ', Message)
    message = re.sub('[^a-zA-Z]', ' ', Message)
    message = message.lower()
    message = message.split()
    message = [word for word in message if not word in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    message = [lem.lemmatize(word) for word in message]
    message = ' '.join(message)
    corpus_test.append(message)
    

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
test_actual_label.columns = ['Category']
test_actual_label['Category'] = test_actual_label['Category'].replace({1: 'spam', 0: 'ham'})
test_predicted_label = predict.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace({1: 'spam', 0: 'ham'})
test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
test_result.head(120)