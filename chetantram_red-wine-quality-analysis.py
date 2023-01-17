import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.preprocessing import LabelEncoder,StandardScaler
data=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv",encoding="ISO-8859-1")
data.head()
data.describe().T
data.info()
data.isnull().sum()
#data=data.fillna(0,inplace=True)
sns.pairplot(data)
data.corr()
plt.figure(figsize=(12,7))

sns.heatmap(data.corr())
data['quality'].unique()
# to check the no. of records in each unique category
from collections import Counter

Counter(data['quality'])
sns.countplot(data['quality'])
#creating new row'Reviews' in df
reviews=[]

for i in data['quality']:

    if i>=1 and i<=3:

        reviews.append('1')

    elif i>=4 and i<=7:

        reviews.append('2')

    elif i>=8 and i<=10:

        reviews.append('3')

data['reviews']=reviews
data['reviews'].unique()
Counter(data['reviews'])
x=data.drop('reviews',axis=1)

y=data['reviews']
#Scaling x and y for PCA

sc=StandardScaler()
x=sc.fit_transform(x)

print(x)
from sklearn.decomposition import PCA

pca=PCA()
xpca=pca.fit_transform(x)
pca.explained_variance_ratio_
#plotting to find principal components
plt.figure(figsize=(10,10))

plt.plot(np.cumsum(pca.explained_variance_ratio_),'-ro')

plt.grid
#we can see that 8 components attribute 90% of variation
pca_new=PCA(n_components=7)

x_new=pca.fit_transform(x)
print(x_new)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
#Logistic Regression

lr=LogisticRegression()

lr.fit(x_train,y_train)

lrpred=lr.predict(x_test)
print(lrpred)
#Decision Trees

lr_conf_matrix = confusion_matrix(y_test, lrpred)

lr_acc_score = accuracy_score(y_test, lrpred)

print(lr_conf_matrix)

print(lr_acc_score*100)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

dtpred = dt.predict(x_test)
print(dtpred)
dt_conf_matrix = confusion_matrix(y_test, dtpred)

dt_acc_score = accuracy_score(y_test, dtpred)

print(dt_conf_matrix)

print(dt_acc_score*100)
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

nbpred=nb.predict(x_test)
print(nbpred)
nb_conf_matrix = confusion_matrix(y_test, nbpred)

nb_acc_score = accuracy_score(y_test, nbpred)

print(nb_conf_matrix)

print(nb_acc_score*100)
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x_train,y_train)

rfpred=rf.predict(x_test)
print(rfpred)
rf_conf_matrix = confusion_matrix(y_test, rfpred)

rf_acc_score = accuracy_score(y_test, rfpred)

print(rf_conf_matrix)

print(rf_acc_score*100)
#Overall all the model predictions

print("Logistic Regression: ",lr_acc_score*100)

print("Decision Trees: ",dt_acc_score*100)

print("NaiveBayes: ",nb_acc_score*100)

print("Random Foresst: ",rf_acc_score*100)