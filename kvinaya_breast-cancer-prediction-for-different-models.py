# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn import svm

from sklearn.model_selection import cross_val_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")

data.head()
data.shape
data.isnull().sum()

data.columns #print the total columns
data=data.drop("id",axis=1)
data=data.drop("Unnamed: 32",axis=1)
data["diagnosis1"]=data["diagnosis"].map({'B':0,'M':1}).astype(int)

data.head()
data.shape
colors=['green',"violet"]

data.diagnosis.value_counts().plot(kind="pie",colors=colors,autopct='%1.1f%%')

plt.axis("equal")

plt.show()
feature_mean=list(data.columns[1:11])

feature_se=list(data.columns[11:21])

feature_worst=list(data.columns[21:31])

print(feature_mean)

print(feature_se)

print(feature_worst)
corr=data.corr()

corr.head()
corr=data[feature_se].corr()

fig=plt.figure(figsize=(10,10))

sns.heatmap(corr,cbar = True, square = True, annot=True,annot_kws={'size': 12}, fmt= '.2f',xticklabels= feature_mean, yticklabels= feature_mean)
corr=data[feature_se].corr()

fig=plt.figure(figsize=(10,10))

sns.heatmap(corr,cbar = True, square = True, annot=True,annot_kws={'size': 16}, fmt= '.2f',xticklabels= feature_se, yticklabels= feature_se)
corr=data[feature_worst].corr()

fig=plt.figure(figsize=(10,10))

sns.heatmap(corr,cbar = True, square = True, annot=True,annot_kws={'size': 16}, fmt= '.2f',linecolor='white',xticklabels= feature_worst, yticklabels= feature_worst)
fig, ax = plt.subplots(nrows=4,ncols=2)

fig.subplots_adjust(right=1.5,top=1.5,wspace = 0.5,hspace = 0.5 )

plt.subplot(4,2,1)

sns.regplot(x='perimeter_mean',y='area_mean',data=data);



plt.subplot(4,2,2)

sns.regplot(x='perimeter_se',y='area_se',data=data);



plt.subplot(4,2,3)

sns.regplot(x='concave points_worst',y='concavity_worst',data=data);



plt.subplot(4,2,4)

sns.regplot(x='radius_worst',y='perimeter_worst',data=data);



plt.subplot(4,2,5)

sns.regplot(x='radius_mean',y='perimeter_worst',data=data);



plt.subplot(4,2,6)

sns.regplot(x='concave points_se',y='concavity_se',data=data);



plt.subplot(4,2,7)

sns.regplot(x='concave points_mean',y='concavity_mean',data=data);



plt.subplot(4,2,8)

sns.regplot(x='compactness_worst',y='concave points_worst',data=data);

plt.show()

from sklearn import preprocessing

fea_mean = pd.DataFrame(preprocessing.scale(data.iloc[:,1:11]))

fea_mean.columns = list(data.iloc[:,1:11].columns)

fea_mean['diagnosis']=data["diagnosis"]

fea_mean.columns
f1 = pd.melt(fea_mean, "diagnosis", var_name="features")

fig, ax = plt.subplots(figsize=(12,8))

p = sns.boxplot(ax = ax, x="features", y="value", hue="diagnosis",data=f1, palette = 'Set3')

p.set_xticklabels(rotation = 90, labels = list(fea_mean.columns));

fea_se = pd.DataFrame(preprocessing.scale(data.iloc[:,11:21]))

fea_se.columns=list(data.iloc[:,11:21].columns)

fea_se["diagnosis"]=data["diagnosis"]

f2 = pd.melt(fea_se, "diagnosis", var_name="features")

fig, ax = plt.subplots(figsize=(12,8))

p = sns.boxplot(ax = ax, x="features", y="value", hue="diagnosis",data=f2, palette = 'Set3');

p.set_xticklabels(rotation = 90, labels = list(fea_se.columns));

fea_worst = pd.DataFrame(preprocessing.scale(data.iloc[:,21:31]))

fea_worst.columns = list(data.iloc[:,21:31].columns)

fea_worst['diagnosis']=data["diagnosis"]

f3=pd.melt(fea_worst,"diagnosis",var_name="features")

fig,ax=plt.subplots(figsize=(12,8))

p=sns.boxplot(ax=ax,x="features",y="value",hue="diagnosis",data=f3,palette='Set3');

p.set_xticklabels(rotation=90,labels=list(fea_worst.columns))

plt.show()
color_function = {0: "green", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis1"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

pd.scatter_matrix(data[feature_mean], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
color_function = {0: "green", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis1"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

pd.scatter_matrix(data[feature_se], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
color_function = {0: "green", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis1"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

pd.scatter_matrix(data[feature_worst], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
x=data[['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se', 'area_se','compactness_se', 'concave points_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','texture_worst','area_worst']]

y=data[['diagnosis']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = svm.SVC(kernel='linear', C=10)

model.fit(x_train, y_train)

predicted= model.predict(x_test)
acc = model.score(x_test,y_test)

acc
acc = model.score(x_train,y_train)

acc
from sklearn.model_selection import cross_val_score

score=cross_val_score(model,x,y,cv=2)

score
model = DecisionTreeClassifier(max_depth=7,random_state=0)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

acc
acc = model.score(x_train,y_train)

acc
score=cross_val_score(model,x,y,cv=2)

score
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=7), n_estimators=5,bootstrap=True, bootstrap_features=False, oob_score=True, random_state=1, verbose=1)

model.fit(x_train, y_train)

pred=model.predict(x_test)

acc = model.score(x_test,y_test)

acc
acc = model.score(x_train,y_train)

acc
score=cross_val_score(model,x,y,cv=2)

score
model=RandomForestClassifier(max_depth=6,random_state=5)

model.fit(x_train,y_train)

predict=model.predict(x_test)

acc = model.score(x_test,y_test)

acc

acc = model.score(x_train,y_train)

acc
score=cross_val_score(model,x,y,cv=2)

score