# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import os



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.chdir('../input')
df = pd.read_csv('heart.csv')
df.head()
df.info()
df.iloc[:,:-1].corr()


df=df.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps',

                      'chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach',

                      'exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca',

                      'thal':'Thal','target':'Target'})
df.isnull().sum()
plt.title('Sex freq', fontsize=25)

df['Sex'].value_counts().plot.pie(figsize=(12,9), explode=(0.05,0), shadow=True, startangle=90, fontsize=15, autopct='%1.1f%%', labels=('Male','Female'))
plt.title('Cp freq', fontsize=25)

plt.axis('equal')

df['Cp'].value_counts().plot.pie(figsize=(12,9), labels=['Type : 0','Type : 1', 'Type : 2', 'Type : 3'], startangle=180, autopct='%1.1f%%', fontsize=15)
plt.title('fasting blood sugar freq', fontsize=25)

plt.axis('equal')

df['Fbs'].value_counts().plot.pie(figsize=(12,9), labels=['fasting blood sugar > 120mg/dL','fasting blood sugar < 120mg/dL'], startangle=180, autopct='%1.1f%%', fontsize=15, explode=(0.1,0))
plt.title('exercise induced angina', fontsize=25)

plt.axis('equal')

df['Exang'].value_counts().plot.pie(figsize=(12,9), labels=['Type : 0','Type : 1'], startangle=90, autopct='%1.1f%%', fontsize=15, explode=(0.1,0))
plt.title('have a Heart disease?', fontsize=25)

plt.axis('equal')

df['Target'].value_counts().plot.pie(figsize=(12,9), labels=['Yes','No'], startangle=90, autopct='%1.1f%%', fontsize=15, explode=(0.1,0))
plt.figure(figsize=(14,8))

sns.heatmap(df.corr(), annot = True, cmap='coolwarm', linewidths=.1)
sns.distplot(df['Thalach'], kde=False, bins=30, color='violet')
sns.distplot(df['Chol'], kde=False, bins=30, color='red')
sns.distplot(df['Trestbps'], kde=False, bins=30, color='blue')
plt.figure(figsize=(8,6))

sns.set(style='whitegrid')

sns.violinplot(df['Target'],df['Age'], hue=df['Cp'], palette='muted')
plt.figure(figsize=(8,6))

sns.scatterplot(x='Chol',y='Thalach',data=df,hue='Target')

plt.show()
X = df.drop('Target', axis=1)

y = df['Target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test= scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}
model = GridSearchCV(knn, params, cv=10)
model.fit(X_train, y_train)
model.best_params_
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy Score: ',accuracy_score(y_test,pred))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,pred),5)*100,'%')
cnt_matrix = confusion_matrix(y_test, pred)

cnt_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))
from sklearn.metrics import roc_auc_score,roc_curve

y_prob = model.predict_proba(X_test)[:,1]

knn_false_pos_rate, knn_true_pos_rate, knn_threshold = roc_curve(y_test, y_prob)
def drawRocCurve(*args):

    a , b, roc_auc_score = args

    plt.figure(figsize=(10,6))

    plt.title('Revceiver Operating Characterstic')

    plt.plot(a,b)

    plt.text(0.8, 0.6, 'roc auc scroe : {}'.format(round(roc_auc_score,3)))

    plt.plot([0,1],ls='--')

    plt.plot([0,0],[1,0],c='.5')

    plt.plot([1,1],c='.5')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.show()
score = roc_auc_score(y_test, y_prob)

drawRocCurve(knn_false_pos_rate, knn_true_pos_rate, score)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
params = {'penalty' : ['l1','l2'],

         'C' : [0.01, 0.1, 1, 10, 100],

         'class_weight' : ['balanced', None]}

log_model = GridSearchCV(log, param_grid=params, cv=10)

log_model.fit(X_train, y_train)

log_model.best_params_
predict = log_model.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using Logistic Regression we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve

print(classification_report(y_test,predict))
cnf_matrix = confusion_matrix(y_test,predict)

cnf_matrix
# get predicted prob..

target_prob_log = log_model.predict_proba(X_test)[:,1]
# create true and false positive rates

log_false_pos_rate, log_true_pos_rate, log_threshold = roc_curve(y_test, target_prob_log)
score = roc_auc_score(y_test, target_prob_log)

drawRocCurve(log_false_pos_rate, log_true_pos_rate, score)
from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier(random_state=42)
param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=4, criterion='gini')
rfc1.fit(X_train, y_train)
predict = rfc1.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using Logistic Regression we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve

print(classification_report(y_test,predict))
cnf_matrix = confusion_matrix(y_test,predict)

cnf_matrix
# get predicted prob..

target_prob_rfc = rfc1.predict_proba(X_test)[:,1]
# create true and false positive rates

rfc_false_pos_rate, rfc_true_pos_rate, rfc_threshold = roc_curve(y_test, target_prob_rfc)
score = roc_auc_score(y_test, target_prob_rfc)

drawRocCurve(rfc_false_pos_rate, rfc_true_pos_rate, score)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(knn_false_pos_rate, knn_true_pos_rate,label='k-Nearest Neighbor')

plt.plot(log_false_pos_rate, log_true_pos_rate,label='Logistic Regression')

plt.plot(rfc_false_pos_rate, rfc_true_pos_rate,label='Random Forest Classifier')

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.legend()

plt.show()