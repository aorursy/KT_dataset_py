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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from string import punctuation

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

df = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')
df.head()
df.isna().sum()
df.info()
df.columns
df.drop(['author_flair_text', 'removed_by',
       'total_awards_received', 'awarders'], axis=1, inplace=True)
del df['id']
del df['created_utc']
del df['full_link']
df.head()
df.title.fillna(" ", inplace=True)
df['text'] = df['title'] + df['author']
del df['title']
del df['author']
def target(val):
    if val == False: return 1
    else: return 0

df['target'] = df['over_18'].apply(target)
del df['over_18']
sns.countplot(df['target'])
x = df[:100000]
train_false = x[x.target ==  1].text
train_true = x[x.target == 0].text
train_text = df.text.values[:100000]
test_text = df.text.values[100000:]
train_category = df.target[:100000]
test_category = df.target[100000:]
plt.figure(figsize = (20,20))
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(train_true)))
plt.imshow(wc,interpolation = 'bilinear')
text_true = wc.process_text(str(" ".join(train_true))) # Getting the most frequently used words from wordcloud 
list(text_true.keys())[:10]
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
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

cleaned_text = []

for sentance in tqdm(df['text'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_text.append(sentance.strip())
df['text'] = cleaned_text
X = df['text']
y = df['target']
from sklearn.model_selection import train_test_split
X_Train, X_test, y_Train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify = y)
X_train, X_cross, y_train, y_cross = train_test_split(X_Train, y_Train, test_size=0.1, random_state=42, stratify = y_Train)
tf_idf=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))
tf_idf.fit(X_train)
Train_TFIDF = tf_idf.transform(X_train)
CrossVal_TFIDF = tf_idf.transform(X_cross)
Test_TFIDF= tf_idf.transform(X_test)
alpha_set=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []


for i in alpha_set:
    naive_b=MultinomialNB(alpha=i)
    naive_b.fit(Train_TFIDF, y_train)
    Train_y_pred =  naive_b.predict(Train_TFIDF)
    Train_AUC_TFIDF.append(roc_auc_score(y_train,Train_y_pred))
    CrossVal_y_pred =  naive_b.predict(CrossVal_TFIDF)
    CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred))
Alpha_set=[]
for i in range(len(alpha_set)):
    Alpha_set.append(np.math.log(alpha_set[i]))
plt.plot(Alpha_set, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(Alpha_set, Train_AUC_TFIDF)
plt.plot(Alpha_set, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(Alpha_set, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_alpha=alpha_set[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(optimal_alpha)
Classifier2 = MultinomialNB(alpha=optimal_alpha)
Classifier2.fit(Train_TFIDF, y_train)
print ("Accuracy is: ", accuracy_score(y_train,Classifier2.predict(Train_TFIDF)))
print ("Accuracy is: ", accuracy_score(y_test,Classifier2.predict(Test_TFIDF)))
print('Confusion Matrix of Train Data')
Train_mat=confusion_matrix(y_test,Classifier2.predict(Test_TFIDF))
print (Train_mat)
from sklearn import metrics
print(metrics.classification_report(y_test,Classifier2.predict(Test_TFIDF)))
print('Confusion Matrix of Train Data')
Train_mat=confusion_matrix(y_train,Classifier2.predict(Train_TFIDF))
print (Train_mat)
plot_confusion_matrix(Classifier2, Test_TFIDF, y_test ,display_labels=['0','1'],cmap="Blues",values_format = '')

c=[0.0001,0.001,0.01,0.1,1,10,100,1000]
Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []
for i in c:
    logreg = LogisticRegression(C=i,penalty='l2')
    logreg.fit(Train_TFIDF, y_train)
    Train_y_pred =  logreg.predict(Train_TFIDF)
    Train_AUC_TFIDF.append(roc_auc_score(y_train ,Train_y_pred))
    CrossVal_y_pred =  logreg.predict(CrossVal_TFIDF)
    CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred))
C=[]
for i in range(len(c)):
    C.append(np.math.log(c[i]))
plt.plot(C, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(C, Train_AUC_TFIDF)
plt.plot(C, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(C, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("lambda : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_inverse_lambda=c[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(pow(optimal_inverse_lambda,-1))
Classifier=LogisticRegression(C=optimal_inverse_lambda,penalty='l2')
Classifier.fit(Train_TFIDF, y_train)
print ("Accuracy is: ", accuracy_score(y_train,Classifier.predict(Train_TFIDF)))
print ("Accuracy is: ", accuracy_score(y_test,Classifier.predict(Test_TFIDF)))
from sklearn import metrics
print(metrics.classification_report(y_test,Classifier.predict(Test_TFIDF)))
plot_confusion_matrix(Classifier, Test_TFIDF, y_test ,display_labels=['0','1'],cmap="Blues",values_format = '')
