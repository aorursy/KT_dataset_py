import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
df=pd.read_csv('../input/heart.csv')
df.head()
df.describe()
df.info()
df.isnull().any()
df['age_cat']=pd.cut(df['age'], 5)
sns.barplot(x=df.groupby(by="age_cat").target.sum().index,y=df.groupby(by="age_cat").target.sum())
sns.catplot(data = df, y ="age", x = "target", hue = 'sex', sharex=False)
sns.countplot(x=df['target'],hue=df['sex'])
df['cp'].unique()
df['cp'] = df['cp'].replace(0, 'typical angina')

df['cp'] = df['cp'].replace(1, 'atypical angina')

df['cp'] = df['cp'].replace(2, 'non-anginal pain')

df['cp'] = df['cp'].replace(3, 'asymptomatic')
sns.barplot(x=df.groupby(by="cp").target.sum().index,y=df.groupby(by="cp").target.sum())
df['trestbps'].unique()
df['bps_cat']=pd.cut(df['trestbps'], 5)
sns.barplot(x=df.groupby(by="bps_cat").target.sum().index,y=df.groupby(by="bps_cat").target.sum())
sns.catplot(data = df, y ="trestbps", x = "target", hue='sex' ,sharex=False)
df['chol_cat']=pd.cut(df['chol'], 5)
sns.barplot(x=df.groupby(by="chol_cat").target.sum().index,y=df.groupby(by="chol_cat").target.sum())
sns.catplot(data = df, y ="chol", x = "target", hue='sex' ,sharex=False)
sns.countplot(data = df, x = "target", hue='fbs')
sns.countplot(data = df, x = "target", hue='restecg')
sns.catplot(data = df, y ="restecg", x = "target" ,sharex=False)
df['thalach_cat']=pd.cut(df['thalach'], 5)
sns.barplot(x=df.groupby(by="thalach_cat").target.sum().index,y=df.groupby(by="thalach_cat").target.sum())
sns.catplot(data = df, y ="thalach", x = "target" ,sharex=False)
sns.countplot(data = df, x = "target", hue='exang')
df['oldpeak_cat']=pd.cut(df['oldpeak'], 5)
sns.barplot(x=df.groupby(by="oldpeak_cat").target.sum().index,y=df.groupby(by="oldpeak_cat").target.sum())
sns.catplot(data = df, y ="oldpeak", x = "target" ,sharex=False)
df['slope'] = df['slope'].replace(0,'upslope')

df['slope'] = df['slope'].replace(1,'flatslope')

df['slope'] = df['slope'].replace(2,'downslope')
sns.countplot(data = df, x = "target", hue='slope')
sns.catplot(data = df, x ="slope", y = "target" ,sharex=False)
sns.countplot(data = df, x = "target", hue='ca')
df['thal'].unique()
sns.countplot(data = df, x = "target", hue='thal')
slope = pd.get_dummies(df['slope'])

slope.drop(slope.columns[[0]],axis=1,inplace=True)

df = pd.concat([df,slope],axis=1)
thal = pd.get_dummies(df['thal'],prefix='thal')

thal.drop(thal.columns[[0]],axis=1,inplace=True)

df = pd.concat([df,thal],axis=1)
del df['age_cat']

del df['bps_cat']

del df['chol_cat']

del df['thalach_cat']

del df['oldpeak_cat']
df.head()
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),annot=True,cmap='Blues')
features = ['thal_3','thal_3','thal_1','flatslope','ca','oldpeak','exang','thalach','restecg','trestbps','sex','age']

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics
def conflog(features):

    X = df[features]

    y = df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    log = LogisticRegression(multi_class='auto')

    log.fit(X_train,y_train)

    y_train_pred = log.predict(X_train)   

    y_test_pred = log.predict(X_test)   

    print('for train')

    print(metrics.classification_report(y_train,y_train_pred))

    print(metrics.accuracy_score(y_train,y_train_pred))

    print('for test')

    print(metrics.classification_report(y_test,y_test_pred))

    print(metrics.accuracy_score(y_test,y_test_pred))
conflog(features)
def confNB(features):

    X = df[features]

    y = df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    nb = GaussianNB()

    nb.fit(X_train,y_train)

    y_train_pred = nb.predict(X_train)   

    y_test_pred = nb.predict(X_test)   

    print('for train')

    print(metrics.classification_report(y_train,y_train_pred))

    print(metrics.accuracy_score(y_train,y_train_pred))

    print('for test')

    print(metrics.classification_report(y_test,y_test_pred))

    print(metrics.accuracy_score(y_test,y_test_pred))
confNB(features)
def confDT(features):

    X = df[features]

    y = df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    dt = DecisionTreeClassifier()

    dt.fit(X_train,y_train)

    y_train_pred = dt.predict(X_train)   

    y_test_pred = dt.predict(X_test)   

    print('for train')

    print(metrics.classification_report(y_train,y_train_pred))

    print(metrics.accuracy_score(y_train,y_train_pred))

    print('for test')

    print(metrics.classification_report(y_test,y_test_pred))

    print(metrics.accuracy_score(y_test,y_test_pred))
confDT(features)
def confSVM(features):

    X = df[features]

    y = df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    svm = SVC(gamma='scale')

    svm.fit(X_train,y_train)

    y_train_pred = svm.predict(X_train)   

    y_test_pred = svm.predict(X_test)   

    print('for train')

    print(metrics.classification_report(y_train,y_train_pred))

    print(metrics.accuracy_score(y_train,y_train_pred))

    print('for test')

    print(metrics.classification_report(y_test,y_test_pred))

    print(metrics.accuracy_score(y_test,y_test_pred))
confSVM(features)
def confRF(features):

    X = df[features]

    y = df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=125)

    rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    rf.fit(X_train,y_train)

    y_train_pred = rf.predict(X_train)   

    y_test_pred = rf.predict(X_test)   

    print('for train')

    print(metrics.classification_report(y_train,y_train_pred))

    print(metrics.accuracy_score(y_train,y_train_pred))

    print('for test')

    print(metrics.classification_report(y_test,y_test_pred))

    print(metrics.accuracy_score(y_test,y_test_pred))
confRF(features)