import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pytz as tz
#from datetime import datetime
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from collections import OrderedDict
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
df1 = pd.concat(pd.read_excel('../input/datasetv631-1/dataset-v6.3(1) (1).xls', sheet_name=None), ignore_index=False)
print(df1.info())
print(df1.shape)
print(df1.isnull().sum())
df1.head(10)
df1 = df1.fillna(0)
for col in df1.columns:
    print(col, ":", df1[col].unique().shape[0])
plt.figure(figsize=(7,8))
df1['O.inherit'].value_counts()[:5].plot(kind='pie',autopct='%1.1f%%',shadow=True,legend = True)
plt.show()
data=df1.iloc[:,[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,47,48,49,50,51,52,53,54,55,58,59,63,64,65,66,67,68,69,71,72,73,74,75,76,77,78]].astype(str)
label_encoders = {}
for column in data:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
    from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
data
df=pd.concat([d0,data], axis=1)
df
plt.figure(figsize=(15,8))
sns.countplot(df1['R.type'])
# Check the top ten countries in the dataset with highest transactions
df1['R.type'].value_counts(normalize=True).head(3).mul(100).round(1).astype(str) + '%'
df0=df.drop(columns='R.type')
df = df0.fillna(0)
df=pd.concat([df1['R.type'],df],axis=1)
#Observing correlation
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
countNoDisease = len(df[df['R.type'] == 0])
countHaveastigmatism = len(df[df['R.type'] == 1])
countHaveastigmatismhyperopia = len(df[df['R.type'] == 2])
countHaveastigmatismmyopia = len(df[df['R.type'] == 3])
countHavehyperopia = len(df[df['R.type'] == 4])
countHavemyopia = len(df[df['R.type'] == 5])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df['R.type']))*100)))
print("Percentage of Patients Have Disease: {:.2f}%".format((countHaveastigmatism / (len(df['R.type']))*100)))
print("Percentage of Patients Have Disease: {:.2f}%".format((countHaveastigmatismhyperopia / (len(df['R.type']))*100)))
print("Percentage of Patients Have Disease: {:.2f}%".format((countHaveastigmatismmyopia / (len(df['R.type']))*100)))
print("Percentage of Patients Have Disease: {:.2f}%".format((countHavehyperopia / (len(df['R.type']))*100)))
print("Percentage of Patients Have Disease: {:.2f}%".format((countHavemyopia / (len(df['R.type']))*100)))
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
X = df.drop(['R.type'],1)   #Feature Matrix
y = df['R.type']         #Target Variable