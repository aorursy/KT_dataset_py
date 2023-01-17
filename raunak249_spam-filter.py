import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/spam.csv',encoding='latin-1')
dataset.head()
dataset = dataset.drop(['Unnamed: 2' , 'Unnamed: 3', 'Unnamed: 4'],axis=1)
dataset.head()
dataset = dataset.rename(columns={'v1':'Target','v2':'SMS'})
import nltk
#We need this to remove the unwanted characters
import re
# The PorterStemmer will give the stem word of each word in the sms column
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
corpus = []
# This is going to keep the cleaned text saved in it


for i in range(len(dataset['SMS'])):
 clean_sms = re.sub('[^a-zA-Z]',' ',dataset['SMS'][i])
 clean_sms = clean_sms.lower()
 clean_sms = clean_sms.split()
 clean_sms = [ps.stem(word) for word in clean_sms if not word in set(stopwords.words('english'))]
 clean_sms = ' '.join(clean_sms)
 corpus.append(clean_sms)

from sklearn.feature_extraction.text import  CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
X
y = dataset.iloc[:,0].values
y

for i in range(len(y)):
    if y[i] == 'ham':
        y[i] = 1
    else:
        y[i] = 0
    
y = y.astype(np.int64)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 2)
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train,y_train)
y_pred = gb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
cn = confusion_matrix(y_test,y_pred)
print(cn)
print(classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200)
rf.fit(X_train,y_train)
y_pred2 = rf.predict(X_test)

cn2 = confusion_matrix(y_test,y_pred2)
print(cn2)

print(classification_report(y_test,y_pred2))
from sklearn.svm import SVC
svc = SVC(kernel = 'sigmoid',gamma = 1.0)
svc.fit(X_train,y_train)
pred3 = svc.predict(X_test)
cn3 = confusion_matrix(y_test,pred3)
cn3
print(classification_report(y_test,pred3))
