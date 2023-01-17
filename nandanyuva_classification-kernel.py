import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

os.listdir("../input")
dsh=pd.read_csv("../input/Hospitaldata.csv")

dsp=pd.read_csv("../input/Patientdata.csv")

dsd=pd.read_csv("../input/Diagnosisdata.csv")
dsh=pd.read_data=pd.read_csv("../input/Hospitaldata.csv")

dsp=pd.read_data=pd.read_csv("../input/Patientdata.csv")

dsd=pd.read_data=pd.read_csv("../input/Diagnosisdata.csv")
dsh.shape
dfh=dsh[dsh['istrain']!= 0]

dfp=dsp[dsp['istrain']!=0]

dfd=dsd[dsd['istrain']!=0]

dfh.istrain.value_counts(),dfd.istrain.value_counts(),dfp.istrain.value_counts()
dfp.isnull().sum(),dfd.isnull().sum(),dfh.isnull().sum(),
y=dfp['Target']

dfp.drop(['weight','Target'],axis=1,inplace = True)

dfh.drop(['payer_code','medical_specialty'],axis=1,inplace = True)
dfd.isna().sum()
dff1=pd.concat([dfh,dfd],axis=1)

dffinal = pd.concat([dff1,dfp],axis =1)
# x=pd.get_dummies(dffinal) #memory error

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 

X=dffinal.apply(label_encoder.fit_transform)

#X= label_encoder.fit_transform(dffinal) 
y.value_counts()
y_replaced= y.replace({'No':0, 'Yes':1})
y_replaced.value_counts()
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB

Dclassifier = DecisionTreeClassifier()

mnbclassifier=MultinomialNB()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y_replaced, test_size=0.20)
mnbclassifier.fit(X_train,y_train)

y_pred_mnb = mnbclassifier.predict(X_test)
Dclassifier.fit(X_train,y_train)

y_pred_dt = Dclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred_dt))
import matplotlib.pyplot as plt

import seaborn as sb
conf_matrix = confusion_matrix(y_test, y_pred_mnb)

sb.heatmap(conf_matrix, annot=True, fmt="d")
#multinomial niave

print(classification_report(y_test, y_pred_mnb))

#very bad because of naive feature 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf_rf = RandomForestClassifier(random_state=23)      

clr_rf = clf_rf.fit(X_train,y_train)

y_predict_rnf = clf_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict_rnf)

print('Accuracy: ', accuracy)

#accuracy got incresed compared to decision tree 
from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics

#count_vectorizer



xg = AdaBoostClassifier()

xg.fit(X_train,y_train)

prediction_xg = xg.predict(X_test)



accuracy_c_v = metrics.accuracy_score(y_test,prediction_xg)

print(accuracy_c_v)







tn,fp,fn,tp = metrics.confusion_matrix(y_test, prediction_xg).ravel()

rec = (tp/(tp+fn)) # recall/sensitivity

pre = (tp/(tp+fp)) # precision

F1_cv_xg = (2*pre*rec)/(pre+rec)

print(F1_cv_xg)
y_pred_proba = xg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data , auc="+str(auc))

plt.legend(loc=4)

plt.show()