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
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import plotly.offline as pyoff
import plotly.graph_objs as go
from nltk.util import ngrams
import re
import unicodedata
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from prettytable import PrettyTable
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
df = pd.read_csv('/kaggle/input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')
df.head()
sns.countplot(df['Rating'])
df.isna().sum()
stop = set(stopwords.words('english'))
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

for sentance in tqdm(df['Review'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_title.append(sentance.strip())
df['Review'] = cleaned_title
df.head()
from wordcloud import WordCloud,STOPWORDS
ratings = [1,2,3,4,5]
plt.ion()

for rating in ratings:
    plt.figure(figsize = (20,20))
    userdf = df[df['Rating'] == rating]
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(userdf.Review))
    plt.imshow(wc , interpolation = 'bilinear')
    plt.title(rating)
    plt.show()
    plt.draw()
    plt.pause(0.001)
    plt.clf()
def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
def plot_bigram(words):    
    bigram_words=(pd.Series(nltk.ngrams(words, 2)).value_counts())[:30]
    bigram_words=pd.DataFrame(bigram_words)
    bigram_words['idx']=bigram_words.index
    bigram_words['idx']=bigram_words.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
    plot_data = [
        go.Bar(
            x=bigram_words['idx'],
            y=bigram_words[0],
            #name='True',
            #x_axis="OTI",
            #y_axis="time",
            marker = dict(
                color = 'Red'
            )
        )
    ]
    plot_layout = go.Layout(
            title='Top 30 bi-grams',
            yaxis_title='Count',
            xaxis_title='bi-gram',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)
reviews = basic_clean(''.join(str(df[df['Rating']==1]['Review'].tolist())))
plot_bigram(reviews)
reviews = basic_clean(''.join(str(df[df['Rating']==5]['Review'].tolist())))
plot_bigram(reviews)
def plot_trigram(words):    
    trigram_words=(pd.Series(nltk.ngrams(words, 3)).value_counts())[:30]
    trigram_words=pd.DataFrame(trigram_words)
    trigram_words['idx']=trigram_words.index
    trigram_words['idx']=trigram_words.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+', ' + x['idx'][2]+')',axis=1)
    plot_data = [
        go.Bar(
            x=trigram_words['idx'],
            y=trigram_words[0],
            #name='True',
            #x_axis="OTI",
            #y_axis="time",
            marker = dict(
                color = 'Green'
            )
        )
    ]
    plot_layout = go.Layout(
            title='Top 30 tri-grams',
            yaxis_title='Count',
            xaxis_title='Tri-gram',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)
reviews = basic_clean(''.join(str(df[df['Rating']==1]['Review'].tolist())))
plot_trigram(reviews)
reviews = basic_clean(''.join(str(df[df['Rating']==5]['Review'].tolist())))
plot_trigram(reviews)
reviews = basic_clean(''.join(str(df[df['Rating']==3]['Review'].tolist())))
plot_bigram(reviews)
reviews = basic_clean(''.join(str(df[df['Rating']==2]['Review'].tolist())))
plot_trigram(reviews)
X = df['Review']
y = df['Rating']

from sklearn.model_selection import train_test_split

X_Train, X_test, y_Train, y_test = train_test_split(X, y, random_state=0, stratify=y, test_size=0.15)
print('Test set created')
X_train, X_cross, y_train, y_cross = train_test_split(X_Train, y_Train, random_state=0, stratify=y_Train, test_size=0.15)
print('Cross val set Created')
from sklearn.naive_bayes import MultinomialNB
tf_idf = TfidfVectorizer(ngram_range=(1,3))
tf_idf.fit(X_train)
Train_TFIDF = tf_idf.transform(X_train)
CrossVal_TFIDF = tf_idf.transform(X_cross)
Test_TFIDF= tf_idf.transform(X_test)
alpha_set=[0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]
Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []
for i in alpha_set:
  naive_b=MultinomialNB(alpha=i)
  naive_b.fit(Train_TFIDF, y_train)
  Train_y_pred =  naive_b.predict_proba(Train_TFIDF)[0:,]
  Train_AUC_TFIDF.append(roc_auc_score(y_train,Train_y_pred, multi_class='ovr'))
  CrossVal_y_pred =  naive_b.predict_proba(CrossVal_TFIDF)[0:,]
  CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred, multi_class='ovr'))
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

Classifier=MultinomialNB(alpha=optimal_alpha)
Classifier.fit(Train_TFIDF, y_train)
auc_train_tfidf = roc_auc_score(y_train,Classifier.predict_proba(Train_TFIDF)[0:,], multi_class='ovr')
print ("AUC for Train set", auc_train_tfidf)

auc_test_tfidf = roc_auc_score(y_test,Classifier.predict_proba(Test_TFIDF)[0:,], multi_class='ovr')
print ("AUC for Test set",auc_test_tfidf)
print('Confusion Matrix of Train Data')
Train_mat=confusion_matrix(y_train,Classifier.predict(Train_TFIDF))
print (Train_mat)
print('Confusion Matrix of Test Data')
Test_mat=confusion_matrix(y_test,Classifier.predict(Test_TFIDF))
print (Test_mat)
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(y_test,Classifier.predict(Test_TFIDF))

f1 = f1_score(y_test,Classifier.predict(Test_TFIDF), average='macro')

print ('Accuracy is : ', acc)
print ('F1 Score is :', f1)
count_vect = CountVectorizer()
count_vect.fit(X_train)
X_train_counts = count_vect.transform(X_train)
X_cross_counts = count_vect.transform(X_cross)
X_test_counts = count_vect.transform(X_test)
alpha_set=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
Train_AUC_BOW = []
CrossVal_AUC_BOW = []
for i in alpha_set:
  naive_b=MultinomialNB(alpha=i)
  naive_b.fit(X_train_counts, y_train)
  Train_y_pred =  naive_b.predict_proba(X_train_counts)[0:,]
  Train_AUC_BOW.append(roc_auc_score(y_train,Train_y_pred, multi_class='ovr'))
  CrossVal_y_pred =  naive_b.predict_proba(X_cross_counts)[0:,]
  CrossVal_AUC_BOW.append(roc_auc_score(y_cross,CrossVal_y_pred, multi_class='ovr'))
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
Classifier1.fit(X_train_counts, y_train)
auc_train_bow = roc_auc_score(y_train,Classifier1.predict_proba(X_train_counts)[0:,], multi_class='ovr')
print ("AUC for Train set", auc_train_bow)

auc_test_bow = roc_auc_score(y_test,Classifier1.predict_proba(X_test_counts)[0:,], multi_class='ovr')
print ("AUC for Test set",auc_test_bow)
print('Confusion Matrix of Test Data')
Test_mat=confusion_matrix(y_test,Classifier1.predict(X_test_counts))
print (Test_mat)
recall = np.diag(Test_mat) / np.sum(Test_mat, axis = 1)
precision = np.diag(Test_mat) / np.sum(Test_mat, axis = 0)
c = np.unique(df['Rating'])

plt.plot(c,precision, label='precision')
plt.scatter(c,precision)
plt.plot(c,recall, label='recall')
plt.scatter(c,recall)
plt.plot(c,2*(precision * recall) / (precision + recall) , label='F1 score')
plt.scatter(c,2*(precision * recall) / (precision + recall))
plt.legend()
plt.xlabel("Categories ")
plt.ylabel("Scores")
plt.title("PLOTS")

plt.show()
y_train.value_counts()
from imblearn.over_sampling import SMOTE
strategy = {1:5000, 2:5000, 3:5000, 4:5000, 5:6541}
smote = SMOTE(sampling_strategy=strategy)

X_sm, y_sm = smote.fit_resample(Train_TFIDF,y_train)
X_test_sm, y_test_sm = smote.fit_resample(Test_TFIDF,y_test)
X_cross_sm, y_cross_sm = smote.fit_resample(CrossVal_TFIDF,y_cross)
y_sm.value_counts()
alpha_set=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
Train_AUC_BOW = []
CrossVal_AUC_BOW = []
for i in alpha_set:
  naive_b=MultinomialNB(alpha=i)
  naive_b.fit(X_sm, y_sm)
  Train_y_pred =  naive_b.predict_proba(X_sm)[0:,]
  Train_AUC_BOW.append(roc_auc_score(y_sm,Train_y_pred, multi_class='ovr'))
  CrossVal_y_pred =  naive_b.predict_proba(X_cross_sm)[0:,]
  CrossVal_AUC_BOW.append(roc_auc_score(y_cross_sm,CrossVal_y_pred, multi_class='ovr'))
plt.plot(Alpha_set, Train_AUC_BOW, label='Train AUC')
plt.scatter(Alpha_set, Train_AUC_BOW)
plt.plot(Alpha_set, CrossVal_AUC_BOW, label='CrossVal AUC')
plt.scatter(Alpha_set, CrossVal_AUC_BOW)
plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_alpha=alpha_set[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(optimal_alpha)

Classifier=MultinomialNB(alpha=optimal_alpha)
Classifier.fit(X_sm, y_sm)
print('Confusion Matrix of Test Data')
Test_mat=confusion_matrix(y_test_sm,Classifier.predict(X_test_sm))
print (Test_mat)
acc = accuracy_score(y_test_sm,Classifier.predict(X_test_sm))
f1 = f1_score(y_test_sm,Classifier.predict(X_test_sm), average='macro')

print ('Accuracy is : ', acc)
print ('F1 Score is :', f1)
auc_train_bow = roc_auc_score(y_sm,Classifier.predict_proba(X_sm)[0:,], multi_class='ovr')
print ("AUC for Train set", auc_train_bow)

auc_test_bow = roc_auc_score(y_test_sm,Classifier.predict_proba(X_test_sm)[0:,], multi_class='ovr')
print ("AUC for Test set",auc_test_bow)
print('Confusion Matrix of Test Data')
Test_mat=confusion_matrix(y_test_sm,Classifier.predict(X_test_sm))
print (Test_mat)
recall = np.diag(Test_mat) / np.sum(Test_mat, axis = 1)
precision = np.diag(Test_mat) / np.sum(Test_mat, axis = 0)

c = np.unique(df['Rating'])

plt.plot(c,precision, label='precision')
plt.scatter(c,precision)
plt.plot(c,recall, label='recall')
plt.scatter(c,recall)
plt.plot(c,2*(precision * recall) / (precision + recall) , label='F1 score')
plt.scatter(c,2*(precision * recall) / (precision + recall))
plt.legend()
plt.xlabel("Categories ")
plt.ylabel("Scores")
plt.title("PLOTS")

plt.show()
from sklearn.ensemble import RandomForestClassifier

# define the model
model = RandomForestClassifier(n_estimators=1000, class_weight='balanced', max_depth=50, bootstrap=True,max_features='sqrt')
model.fit(Train_TFIDF, y_train)
y_pred = model.predict(Test_TFIDF)
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(y_train, model.predict(Train_TFIDF)))
# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))


# Classification report
print("Classification report")
print(classification_report(y_test, y_pred))
