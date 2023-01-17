import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from prettytable import PrettyTable
df = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv',  engine = 'python')
df.head()
del df['job_id']
del df['salary_range']
df.fillna(" ",inplace = True)
df['textdata'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function'] 
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['employment_type']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']
df.head()
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
len(stop)
import re,string,unicodedata
punctuation = list(string.punctuation)
stop.update(punctuation)
def cleaner(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", 'can not', phrase)
  
  # general
    phrase = re.sub(r"n\'t"," not", phrase)
    phrase = re.sub(r"\'re'"," are", phrase)
    phrase = re.sub(r"\'s"," is", phrase)
    phrase = re.sub(r"\'ll"," will", phrase)
    phrase = re.sub(r"\'d"," would", phrase)
    phrase = re.sub(r"\'t"," not", phrase)
    phrase = re.sub(r"\'ve"," have", phrase)
    phrase = re.sub(r"\'m"," am", phrase)
    
    return phrase
cleaned_title = []

for sentance in tqdm(df['textdata'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_title.append(sentance.strip())
df['textdata'] = cleaned_title
df.head()
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize = (20,20)) # Text that is not fraudulent(0)
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df[df.fraudulent == 0].textdata))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is  fraudulent(1)
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df[df.fraudulent == 1].textdata))
plt.imshow(wc , interpolation = 'bilinear')
import seaborn as sns
sns.countplot(x = "fraudulent", data=df)
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
X_Train, X_test, y_Train, y_test = train_test_split(df.textdata, df.fraudulent, random_state=0, stratify=df.fraudulent, test_size=0.1)
X_Train.shape
X_train, X_cross, y_train, y_cross = train_test_split(X_Train, y_Train, random_state=0, stratify=y_Train, test_size=0.1)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
y_train.shape
#X_train.shape
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(X_train)
#transformed test reviews

tv_cross_reviews=tv.transform(X_cross)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_cross_reviews.shape)
tv_test_reviews=tv.transform(X_test)
alpha_set=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
Train_AUC_BOW = []
CrossVal_AUC_BOW = []
for i in alpha_set:
    naive_b=MultinomialNB(alpha=i)
    naive_b.fit(tv_train_reviews, y_train)
    Train_y_pred =  naive_b.predict(tv_train_reviews)
    Train_AUC_BOW.append(roc_auc_score(y_train,Train_y_pred))
    CrossVal_y_pred =  naive_b.predict(tv_cross_reviews)
    CrossVal_AUC_BOW.append(roc_auc_score(y_cross,CrossVal_y_pred))
from numpy import math
Alpha_set=[]
for i in range(len(alpha_set)):
    Alpha_set.append(math.log(alpha_set[i]))
plt.plot(Alpha_set, Train_AUC_BOW, label='Train AUC')
plt.scatter(Alpha_set, Train_AUC_BOW)
plt.plot(Alpha_set, CrossVal_AUC_BOW, label='CrossVal AUC')
plt.scatter(Alpha_set, CrossVal_AUC_BOW)
plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_alpha=alpha_set[CrossVal_AUC_BOW.index(max(CrossVal_AUC_BOW))]
print(optimal_alpha)
Classifier1=MultinomialNB(alpha=optimal_alpha)
Classifier1.fit(tv_train_reviews, y_train)
auc_train_bow = roc_auc_score(y_train,Classifier1.predict(tv_train_reviews))
print ("AUC for Train set", auc_train_bow)

auc_test_bow = roc_auc_score(y_test,Classifier1.predict(tv_test_reviews))
print ("AUC for Test set",auc_test_bow)
from sklearn.metrics import accuracy_score, log_loss, f1_score

preds = Classifier1.predict(tv_test_reviews)

acc = accuracy_score(y_test, preds)

f1 = f1_score(y_test, preds, average='macro')

print ('Accuracy is : ', acc)
print ('F1 Score is :', f1)
print('Confusion Matrix of Test Data')
Test_mat=confusion_matrix(y_test,preds)
print (Test_mat)
from sklearn import metrics
print(metrics.classification_report(y_test, preds))
cm_cv = pd.DataFrame(Test_mat, index=[0,1], columns=[0,1])
cm_cv.index.name = 'Actual'
cm_cv.columns.name = 'Predicted'
plt.figure(figsize = (10,10))
sns.heatmap(cm_cv,cmap= "Blues",annot = True )

