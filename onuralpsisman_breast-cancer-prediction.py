# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")
df.shape
df.head()
df.dtypes
df.isnull().sum()
def histogram(variable):

    plt.figure(figsize=(8,4))

    plt.hist(df[variable])

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} frequency with histogram".format(variable))

    plt.show()
df.columns
variables=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',

       'mean_smoothness']

for i in variables:

    histogram(i)

# mean_radius-diagnosis

g= sns.FacetGrid(df,col="diagnosis")

g.map(sns.distplot,"mean_radius",bins=25)
# mean_texture-diagnosis

g= sns.FacetGrid(df,col="diagnosis")

g.map(sns.distplot,"mean_texture",bins=25)
# mean_perimeter-diagnosis

g= sns.FacetGrid(df,col="diagnosis")

g.map(sns.distplot,"mean_perimeter",bins=25)
# mean_area-diagnosis

g= sns.FacetGrid(df,col="diagnosis")

g.map(sns.distplot,"mean_area",bins=25)
# mean_smoothness-diagnosis

g= sns.FacetGrid(df,col="diagnosis")

g.map(sns.distplot,"mean_smoothness",bins=25)
def boxplot(variable):

    plt.subplots()

    plt.boxplot(df[variable])

    plt.xlabel(variable)
for i in variables:

    boxplot(i)
df.drop(df[df["mean_radius"]>25].index,axis=0,inplace=True)

df.drop(df[df["mean_texture"]>35].index,axis=0,inplace=True)

df.drop(df[df["mean_perimeter"]>180].index,axis=0,inplace=True)

df.drop(df[df["mean_area"]>2000].index,axis=0,inplace=True)

df.drop(df[df["mean_smoothness"]>0.15].index,axis=0,inplace=True)
df.shape
def scatter(data,x,y):

    sns.scatterplot(x=x, y=y, data=data, hue = 'diagnosis')
for i in variables:

    for j in variables:

        if i==j:

            pass

        else:

            plt.subplots()

            scatter(df,i,j)
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
x=df.drop(["diagnosis"],axis=1)

y=df["diagnosis"]
scaler=MinMaxScaler()

x_scaled=scaler.fit_transform(x)
random_state=42

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state=random_state)
Confusion_Matrices=[]

Classifiers=[]

Scores=[]



dtc=DecisionTreeClassifier(random_state = random_state)

dtc.fit(x_train,y_train)

Scores.append(cross_val_score(dtc, x_test, y_test, cv=5).mean())

Confusion_Matrices.append(confusion_matrix(y_test, dtc.predict(x_test)))

Classifiers.append("Dtc")
svc=SVC(random_state = random_state)

svc.fit(x_train,y_train)

Scores.append(cross_val_score(svc, x_test, y_test, cv=5).mean())

Confusion_Matrices.append(confusion_matrix(y_test, svc.predict(x_test)))

Classifiers.append("Svc")
for i in [20,50,80,100]:

    rf=RandomForestClassifier(n_estimators=i,random_state = random_state)

    rf.fit(x_train,y_train)

    Scores.append(cross_val_score(rf, x_test, y_test, cv=5).mean())

    Confusion_Matrices.append(confusion_matrix(y_test, rf.predict(x_test)))

    Classifiers.append("Rfc{}".format(str(i)))
lr=LogisticRegression(random_state = random_state)

lr.fit(x_train,y_train)

Scores.append(cross_val_score(lr, x_test, y_test, cv=5).mean())

Confusion_Matrices.append(confusion_matrix(y_test, lr.predict(x_test)))

Classifiers.append("Lr")
for i in [5,6,7,8,9,10]:

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    Scores.append(cross_val_score(knn, x_test, y_test, cv=5).mean())

    Confusion_Matrices.append(confusion_matrix(y_test, knn.predict(x_test)))

    Classifiers.append("Knn{}".format(str(i)))
graph_data= pd.DataFrame(list(zip(Classifiers,Scores)),columns =['Classifiers', 'Scores']) 

graph_data=graph_data.sort_values("Scores",ascending=False)

plt.figure(figsize=(16,8))

sns.barplot(x=graph_data["Classifiers"],y=graph_data["Scores"])

for i in np.arange(0,13):

    plt.subplots()

    sns.heatmap(Confusion_Matrices[i],annot=True)

    plt.title("Confusion Matrix of {}".format(Classifiers[i]))
Ensemble_Model = VotingClassifier(estimators=[('knn6', KNeighborsClassifier(n_neighbors=6)), ('knn5', KNeighborsClassifier(n_neighbors=5)), ('knn10', KNeighborsClassifier(n_neighbors=10)),("svc",SVC(random_state = 42))], voting='hard')
Ensemble_Model.fit(x_train,y_train)
print(cross_val_score(Ensemble_Model, x_test, y_test, cv=5).mean())