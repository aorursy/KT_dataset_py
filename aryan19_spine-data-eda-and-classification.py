import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)  

import cufflinks as cf  

cf.go_offline()  

%matplotlib inline

import os

df = pd.read_csv('../input/Dataset_spine.csv')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.head()
Cols = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope',

       'Direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope']

df = df.rename(index=str,columns={'Col1':'pelvic_incidence','Col2':'pelvic_tilt','Col3':'lumbar_lordosis_angle','Col4':'sacral_slope','Col5':'pelvic_radius','Col6':'degree_spondylolisthesis','Col7':'pelvic_slope','Col8':'Direct_tilt',

                                  'Col9':'thoracic_slope','Col10':'cervical_tilt','Col11':'sacrum_angle','Col12':'scoliosis_slope'})
df.describe()
df.info()
for i in Cols:

    plt.figure(figsize=(20,7))

    df[i].plot(subplots=True)

    plt.title(i)

    plt.show()
A = df['Class_att'].value_counts().index.tolist()

Vals = list(pd.value_counts(df['Class_att']))

explode = [0.1]*len(A)

plt.figure(figsize=(10,10))

plt.pie(Vals,explode=explode,labels=A,autopct='%.1f%%')

plt.show()
sns.pairplot(df)
sns.pairplot(hue='Class_att',data=df)
df.iplot(kind='box')  
from sklearn.preprocessing import LabelEncoder,scale

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report,roc_curve,roc_auc_score,confusion_matrix

from sklearn.model_selection import train_test_split as t
x = df.iloc[:,:-1].values 

y = df.iloc[:,-1].values  #Target Variable
le = LabelEncoder()

y = le.fit_transform(y)
train_x,test_x,train_y,test_y = t(x,y,test_size=0.2)
rfc = RandomForestClassifier(n_estimators=30,max_depth=2)

rfc.fit(train_x, train_y)

score_randomforest = rfc.score(test_x,test_y)

print('The accuracy of the Random Forest Model is', score_randomforest)
df_cm  = pd.DataFrame(confusion_matrix(test_y,rfc.predict(test_x)),['Abnormal','Normal'],['Abormal','Normal'])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)
print(classification_report(test_y,rfc.predict(test_x)))
probs = rfc.predict_proba(test_x)

probs = probs[:,1]

auc = roc_auc_score(test_y,probs)

print('AUC = %.3f' % auc)

fpr,tpr,thresholds = roc_curve(test_y,probs)

plt.plot([0,1],[0,1],linestyle='--')

plt.plot(fpr,tpr,marker='.')

plt.show()