import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
voice=pd.read_csv('../input/voice.csv')
sns.pairplot(voice[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label', 'meanfun', 'IQR']], 
                 hue='label', size=2)
#from this graph we deduce that a good precision can be reached by using only
#2 parameters: meanfun and IQR, because in their graph we can see that the male and female dots are quite
#seperated
#convert labels from string to 0/1
from sklearn.preprocessing import LabelEncoder
gender=LabelEncoder()
y=voice.iloc[:,-1]
y=gender.fit_transform(y)
X=voice.iloc[:,:-1]
y
#standartizetion the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
#logistic reggression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
#now we gonna use svc model
from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(X_train,y_train)
predictions_svc=model_svc.predict(X_test)
print(classification_report(y_test,predictions_svc))
print(confusion_matrix(y_test,predictions_svc))