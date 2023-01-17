
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
drugs = pd.read_csv("/kaggle/input/medicine-data/train.csv")
drugs.head()

drugs.describe()
drugs.dtypes
drugs.shape
drugs[drugs.drugName.duplicated()]
drugs.count()
drugs.isnull().sum()
drugs['condition'].mode()
print("Missing value (%):", 1200/drugs.shape[0] *100)
drugs=drugs.dropna() 
drugs.count()
len(set(drugs['Id'].values))
drugs['Id'].unique()

drugs['drugName'].value_counts()
drugs['output'].value_counts()
import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    i= re.sub("[\W+""]", " ",i)        
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

drugs.review= drugs.review.apply(cleaning_text)
drugs.review
drugs.isnull().sum()
drugs.condition=drugs.condition.apply(cleaning_text)
drugs.condition
import matplotlib.pyplot as plt
import seaborn as sns 
total_drugs = drugs['drugName'].value_counts()
fig = plt.figure(figsize = (60,20))
sns.countplot(total_drugs)
star_rating = drugs['rating'].value_counts()
sns.countplot(star_rating)
conditions = drugs['condition'].value_counts()
fig = plt.figure(figsize = (60,20))
sns.countplot(conditions)
drugs['usefulCount'].idxmax()
drugs.iloc[drugs['usefulCount'].idxmax()]
drugs['review'][drugs['usefulCount'].idxmax()]
drugs[drugs['Id'] == 18004]['review']
plt.plot(drugs.rating,drugs.drugName,"bo");plt.xlabel("Rating");plt.ylabel("Drug name")
plt.show()
plt.plot(drugs.output,drugs.usefulCount,"bo");plt.xlabel("Output");plt.ylabel("UsefulCount")
plt.show()
plt.plot(drugs.output,drugs.condition,"bo");plt.xlabel("OUTPUT");plt.ylabel("Drug_Condition")

sns.pairplot(data=drugs)
word=drugs.review.str.split(expand=True).stack().value_counts()
word
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk ## preprocessing text
from nltk.corpus import stopwords
#stemming & lemmatization help reduce words
#to a common base form or root word
from nltk.tokenize import word_tokenize
sample_words1 = " ".join(drugs.review)
sample_words = sample_words1.split(" ") 
def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist
sample_words=unique_list(sample_words)
from collections import Counter 
# wordcloud in python
from wordcloud import WordCloud, STOPWORDS 
import string
from textblob import TextBlob
stopwords = nltk.corpus.stopwords.words('english')
stopwords
stopwords1 = (w for w in sample_words  if not w in stopwords)
sample_words_string = " ".join(stopwords1)
wordcloud = WordCloud(width=1600, height=1400, random_state=1, max_words=500, background_color='red',)
wordcloud.generate(sample_words_string)
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Top Reviewed words", fontsize=10,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=5)
plt.show()
with open('/kaggle/input/sentiment-dictionary/positive-words.txt',"r") as pos:
  poswords = pos.read().split("\n")
  
poswords1 = poswords[36:]
positive_word= " ".join(w for w in sample_words if w in poswords)
wordcloud = WordCloud(width=1600, height=1400, random_state=1, max_words=500, background_color='yellow',)
wordcloud.generate(positive_word)
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Top Reviewed words", fontsize=10,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=5)
plt.show()
with open('/kaggle/input/sentiment-dictionary/negative-words.txt',encoding = "ISO-8859-1") as neg:
  negwords = neg.read().split("\n")
negwords = negwords[37:]
negative_word = " ".join ([w for w in sample_words if w in negwords])
wordcloud = WordCloud(width=1600, height=1400, random_state=1, max_words=500, background_color='lightblue',)
wordcloud.generate(negative_word)
# declare our figure 
plt.figure(figsize=(20,10))
plt.title("Top Reviewed words", fontsize=10,color='Red')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=5)
plt.show()
word_bag=sample_words
positive_word_bag=positive_word
negative_word_bag=negative_word
import datetime as dt
from dateutil.relativedelta import relativedelta as rd
from datetime import date
drug = pd.DataFrame()
drug['start date']= pd.to_datetime(drugs['date']).dt.date

drug['end date']= pd.to_datetime(date.today())

drug['end date']=drug['end date'].dt.date

drug['days'] =drug['end date']-drug['start date']

drug['days']=drug['days'].dt.days

drugs['days']=drug['days']
drugs
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
drugs['drugName_ID'] = lb_make.fit_transform(drugs['drugName'])

drugs.head() #Results in append
# we will use label encoder on condition too.
drugs['drug_condi'] = lb_make.fit_transform(drugs['condition'])

drugs.head(10) #Results in appendi
from collections import Counter
from nltk import word_tokenize
df2=pd.DataFrame()

df2['review']=drugs.iloc[:,3]
n = set(negwords)
p = set(poswords1)

df2['count'] = df2.iloc[:,0].apply(lambda review: sum(0 + ((word in p) and 1) + ((word in n) and -1) for word in review.split()))
drugs['word_count']=df2.iloc[:,1]
drugs['word_count']
drugdata=drugs[["output","drugName_ID","drug_condi","word_count","days","usefulCount"]]
drugdata.output=drugdata.iloc[:,0].map({'Yes': 0, 'No': 1})
drugdata
sns.pairplot(data=drugdata)
drugdata.corr()
from imblearn.over_sampling import SMOTE
X=drugdata.iloc[:,[3,5]]
Y=drugdata.iloc[:,0]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
sm = SMOTE(random_state = 2) 
X_train_oversampled,Y_train_oversampled = sm.fit_sample(X_train, Y_train)
X_train_oversampled.shape
Y_train_oversampled.shape
traindata=drugdata.iloc[:,[3,5]]
test = pd.read_csv('../input/medicine-data/test.csv')
test['reviewcount'] = test.iloc[:,3].apply(lambda review: sum(0 + ((word in p) and 1) + ((word in n) and -1) for word in review.split()))
test['reviewcount']
testdata=test.iloc[:,[5,6]]
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
LR_model=classifier.fit(X_train_oversampled,Y_train_oversampled)
Y_train_pred = LR_model.predict(traindata)
Y_test_pred = LR_model.predict(testdata)
from sklearn import metrics
LR_train=metrics.accuracy_score(Y,Y_train_pred)
LR_train
result=pd.DataFrame(columns=['Id','output'])
result['Id']=test['Id']
result['output']=Y_test_pred 
result['output']=result['output'].map({0: 'No',1:'Yes'})
result.to_csv('LR_result.csv',header=True)
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_oversampled,Y_train_oversampled)
Y_train_pred = knn.predict(traindata)
KNN_test_pred = knn.predict(testdata)
KNN_train=metrics.accuracy_score(Y,Y_train_pred)
KNN_train
result2=pd.DataFrame(columns=['Id','output'])
result2['Id']=test['Id']
result2['output']=KNN_test_pred
result2['output']=result2['output'].map({0: 'No',1:'Yes'})
result2.to_csv('KNN_result.csv',header=True)
confusion_matrix = pd.crosstab(Y,Y_train_pred)
print (confusion_matrix)
print(classification_report(Y,Y_train_pred))
from sklearn.tree import DecisionTreeClassifier
# create a classifier object 
seed=4
decision_tree_cl = DecisionTreeClassifier(random_state = seed)  
# fit the regressor with X and Y data 
DT_model=decision_tree_cl.fit(X_train_oversampled,Y_train_oversampled) 
Y_DT_train_pred = DT_model.predict(traindata)
DT_test_pred = decision_tree_cl.predict(testdata)
DT_train=metrics.accuracy_score(Y,Y_DT_train_pred)
DT_train
result3=pd.DataFrame(columns=['Id','output'])
result3['Id']=test['Id']
result3['output']=DT_test_pred
result3['output']=result3['output'].map({0: 'No',1:'Yes'})
result3.to_csv('DT_result.csv',header=True)
confusion_matrix = pd.crosstab(Y,Y_DT_train_pred)
print (confusion_matrix)
print(classification_report(Y,Y_DT_train_pred))
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
Random_clf = RandomForestClassifier(n_estimators=150)
Random_model = Random_clf.fit(X_train_oversampled,Y_train_oversampled)
Y_RF_train_pred = Random_model.predict(traindata)
RF_test_pred = Random_model.predict(testdata)
RF_train=metrics.accuracy_score(Y,Y_RF_train_pred)
RF_train
result4=pd.DataFrame(columns=['Id','output'])
result4['Id']=test['Id']
result4['output']=RF_test_pred
result4['output']=result4['output'].map({0: 'No',1:'Yes'})
result4.to_csv('RF_result.csv',header=True)
confusion_matrix = pd.crosstab(Y,Y_RF_train_pred)
print (confusion_matrix)
print(classification_report(Y,Y_RF_train_pred))
from sklearn.svm import SVC
# Kernel = linear
lineaar_svc=SVC(kernel="linear")
lineaar_svc.fit(X_train_oversampled,Y_train_oversampled)
lineaar_svc_pred=lineaar_svc.predict(traindata)
svc_test_pred = lineaar_svc.predict(testdata)
SVC_train=metrics.accuracy_score(Y,lineaar_svc_pred)
SVC_train
result5=pd.DataFrame(columns=['Id','output'])
result5['Id']=test['Id']
result5['output']=svc_test_pred
result5['output']=result5['output'].map({0: 'No',1:'Yes'})
result5.to_csv('linear_SVC_result.csv',header=True)
confusion_matrix = pd.crosstab(Y,lineaar_svc_pred)
print (confusion_matrix)
print(classification_report(Y,lineaar_svc_pred))
from sklearn.svm import SVC
poly_Model=SVC(kernel="poly")
poly_Model.fit(X_train_oversampled,Y_train_oversampled)

poly_Model_pred=poly_Model.predict(traindata)
poly_test_pred = lineaar_svc.predict(testdata)
poly_train=metrics.accuracy_score(Y,poly_Model_pred)
poly_train


from sklearn.svm import SVC
poly_Model=SVC(kernel="poly")
poly_Model.fit(X_train_oversampled,Y_train_oversampled)
poly_Model_pred=poly_Model.predict(X_test)
poly_test_accu=np.mean(poly_Model_pred==Y_test)
pd.crosstab(Y_test_pred,Y_test)
print(classification_report(Y_test,Y_test_pred))
#Kernel = rbf
rbf_Model=SVC(kernel="rbf")
rbf_Model.fit(X_train_oversampled,Y_train_oversampled)
rbf_Model_pred=rbf_Model.predict(X_test)
rbf_test_accu=np.mean(rbf_Model_pred==Y_test)
pd.crosstab(Y_test_pred,Y_test)
print(classification_report(Y_test,Y_test_pred))
#kernel= sigmoid
sigmoid=SVC(kernel="sigmoid")
sigmoid.fit(X_train_oversampled,Y_train_oversampled)
sigmoid_pred=sigmoid.predict(X_test)
sig_test_accu=np.mean(sigmoid_pred==Y_test)#0.37408577127659576,37%
pd.crosstab(Y_test_pred,Y_test)
print(classification_report(Y_test,Y_test_pred))



