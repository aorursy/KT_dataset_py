## Importing the packeges

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Importing the Dataset

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.data
# Forming the DataFrame

cancer_df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

cancer_df['target']=cancer['target']
## Pairplot 

sns.pairplot(cancer_df,vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness']

             ,hue='target')
sns.countplot(cancer_df['target'],label='Count')
## Scatter Plot 

plt.figure(figsize=(8,8))

plot=sns.scatterplot(x='mean radius',y='mean texture',data=cancer_df,hue='target')



plt.figure(figsize=(30,20))

sns.heatmap(cancer_df.corr(),annot=True)
## Differenciating the Variables

X = cancer_df.drop(['target'],axis=1)

y= cancer_df['target']
## Splitting the dataset into train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC

svc_classifier = SVC()

svc_classifier.fit(X_train,y_train)

y_pred = svc_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
print(classification_report(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(0,1))

X_train = mms.fit_transform(X_train)

X_test = mms.fit_transform(X_test)
svc_classifier.fit(X_train,y_train)

y_pred=svc_classifier.predict(X_test)



from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
print(classification_report(y_test, y_pred))