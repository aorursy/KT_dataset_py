import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data
data.info()
data["class"].unique()
data.isnull().any()
data.describe().T
data["class"].value_counts()
sns.countplot(y="class", data=data, palette="hls")

plt.show()
def localSubplot(data,feature):

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))

    

    data[feature].value_counts().plot(kind="bar",ax=ax[0])

    data[feature].value_counts().plot.pie(autopct="%1.1f%%",ax=ax[1])

    

    plt.tight_layout()

    plt.show()
localSubplot(data=data,feature="class")
data.hist(bins=10, density=True, figsize= (12,8))

plt.show()
fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

ax = ax.flatten()

col_names = data.drop('class',axis=1).columns.values



for i,col_name in enumerate(col_names):

    sns.distplot(a=data[col_name], ax=ax[i])
corr= data.corr()

fig, ax=plt.subplots(1,1,figsize=(12,8))

sns.heatmap(corr, annot=True, linewidth=5, ax=ax);
data["class"] = [ 1 if each == "Abnormal" else 0 for each in data["class"]]
X = data.loc[:, data.columns != 'class']

y = data.loc[:, data.columns == 'class']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

lg = LogisticRegression().fit(X_train,y_train)
print("Accuracy: ",lg.score( X_test,y_test)*100)
y_pred = lg.predict(X)

print(classification_report(y,y_pred))
y_pred = lg.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred)*100)