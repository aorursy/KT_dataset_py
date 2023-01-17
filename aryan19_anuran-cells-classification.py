import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split as t

import sklearn.metrics as mt

%matplotlib inline

df = pd.read_csv('../input/Frogs_MFCCs.csv')

df.head()
df = df.drop('RecordID',axis=1)
C1 = df['Family'].values                   #Family

C2 = df['Genus'].values                    #Genus

C3 = df['Species'].values                  #Species

x = df.select_dtypes('float64').values     #Rest of the features
le = LabelEncoder()

C1 = le.fit_transform(C1)

C2 = le.fit_transform(C2)

C3 = le.fit_transform(C3)
train_x1,test_x1,train_y1,test_y1 = t(x,C1,test_size=0.2)  #Family

train_x2,test_x2,train_y2,test_y2 = t(x,C2,test_size=0.2)  #Genus

train_x3,test_x3,train_y3,test_y3 = t(x,C3,test_size=0.2)  #Species
classifiers1 = [

    DecisionTreeClassifier(),

    XGBClassifier(),

    MLPClassifier(hidden_layer_sizes=(100,20,4), max_iter=500),    #4 Subclasses in Family

    SVC(kernel='linear')

    ]



classifiers2 = [

    DecisionTreeClassifier(),

    XGBClassifier(),

    MLPClassifier(hidden_layer_sizes=(100,20,8), max_iter=500),    #8 Subclasses in Genus

    SVC(kernel='linear')

    ]



classifiers3 = [

    DecisionTreeClassifier(),

    XGBClassifier(),

    MLPClassifier(hidden_layer_sizes=(100,20,10), max_iter=500),   #10 Sublclasses in Species

    SVC(kernel='linear')

    ]
Family = df['Family'].value_counts().index.tolist()

Genus = df['Genus'].value_counts().index.tolist()

Species = df['Species'].value_counts().index.tolist()
for c in classifiers1:

    c.fit(train_x1,train_y1)

    df_cm = pd.DataFrame(mt.confusion_matrix(test_y1,c.predict(test_x1)),Family,Family)

    plt.figure(figsize = (10,7))

    plt.title(f'{c.__class__.__name__} Score = {c.score(test_x1,test_y1)}')

    if c==SVC(kernel='linear'):

        print(f'{c.__class__.__name__} Score = {mt.r2_score(test_y1,c.predict(test_x1))}')

    sns.heatmap(df_cm, annot=True)
for c in classifiers1:

    print('='*20+' '+c.__class__.__name__+' '+'='*20)

    print(mt.classification_report(test_y1,c.predict(test_x1)))

    print('\n')
for c in classifiers2:

    c.fit(train_x2,train_y2)

    df_cm = pd.DataFrame(mt.confusion_matrix(test_y2,c.predict(test_x2)),Genus,Genus)

    plt.figure(figsize = (10,7))

    plt.title(f'{c.__class__.__name__} Score = {c.score(test_x2,test_y2)}')

    if c==SVC(kernel='linear'):

        print(f'{c.__class__.__name__} Score = {mt.r2_score(test_y2,c.predict(test_x2))}')

    sns.heatmap(df_cm, annot=True)
for c in classifiers2:

    print('='*20+' '+c.__class__.__name__+' '+'='*20)

    print(mt.classification_report(test_y2,c.predict(test_x2)))

    print('\n')
for c in classifiers3:

    c.fit(train_x3,train_y3)

    df_cm = pd.DataFrame(mt.confusion_matrix(test_y3,c.predict(test_x3)),Species,Species)

    plt.figure(figsize = (10,7))

    plt.title(f'{c.__class__.__name__} Score = {c.score(test_x3,test_y3)}')

    if c==SVC(kernel='linear'):

        print(f'{c.__class__.__name__} Score = {mt.r2_score(test_y3,c.predict(test_x3))}')

    sns.heatmap(df_cm, annot=True)
for c in classifiers3:

    print('='*20+' '+c.__class__.__name__+' '+'='*20)

    print(mt.classification_report(test_y3,c.predict(test_x3)))

    print('\n')