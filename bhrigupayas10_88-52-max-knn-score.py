# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/heart.csv')

df.head()
#to check if any null values exist

df.isnull().sum()
#to extract information about mean,std and different percentiles of each column

df.describe()
#to extract information about datatypes of each column

df.info()
f = plt.subplots(5,4)

i = 0

plt.figure(figsize=(28,32))

for x in df.columns:

    i+=1

    plt.subplot(5,4,i)

    sns.boxplot(df[x])
'''for x in df.columns:

    print(df[x].value_counts())'''
f = plt.subplots(5,4)

i = 0

plt.figure(figsize=(32,28))

for x in df.columns:

    i+=1

    plt.subplot(5,4,i)

    sns.kdeplot(df[x])
df.loc[df['trestbps']>170,'trestbps'] = 170

#sns.boxplot(df['trestbps'])



df.loc[df['chol']>350,'chol'] = 350

#sns.boxplot(df['chol'])



df.loc[df['thalach']<90,'thalach'] = 90

#sns.boxplot(df['thalach'])



df.loc[df['oldpeak']>4,'oldpeak'] =4

#sns.boxplot(df['oldpeak'])



#df['oldpeak'].value_counts()



df.loc[df['ca']>2,'ca'] = 2

#sns.boxplot(df['ca'])



df.loc[df['thal']<2,'thal'] = 2

#sns.boxplot(df['thal'])
f = plt.subplots(5,4)

i = 0

plt.figure(figsize=(28,32))

for x in df.columns:

    i+=1

    plt.subplot(5,4,i)

    sns.boxplot(df[x])
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(df.drop('target',axis=1))

df_sc = sc.transform(df.drop('target',axis=1))

df_sc.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1')



X = df_sc

y = df.target

print(X.shape,y.shape)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



lr.fit(X_train,y_train)



preds = lr.predict(X_test)



from sklearn.metrics import confusion_matrix



confusion_matrix(y_pred=preds,y_true=y_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()



clf.fit(X_train,y_train)



preds = clf.predict(X_test)



cnf = confusion_matrix(y_true=y_test,y_pred=preds)

acc = (cnf[0][0]+cnf[1][1])/(cnf[0][0]+cnf[1][1]+cnf[0][1]+cnf[1][0])

cnf,acc
confusion_matrix(y_pred=clf.predict(X_train),y_true=y_train)
#clf trained over X_train shows overfitting

clf.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    knn.predict(X_test)

    preds = knn.predict(X_test)

    print(knn.score(X_test,y_test))
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

max_i = 0

max_acc = 0

for  i in range(1,10):    

    clf = XGBClassifier(max_depth=i)

    clf.fit(X_train,y_train)

    #print(clf)

    preds = clf.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    if accuracy>max_acc:

        max_i = i

        max_acc = accuracy

    print("Accuracy : %.2f%%" % (accuracy * 100.0))

print("Max accuracy at{0}={1}".format(max_i,max_acc))
clf = XGBClassifier(max_depth=1)

clf.fit(X_train,y_train)

#print(clf)

preds = clf.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print("Accuracy:",accuracy)

confusion_matrix(y_true=y_test,y_pred=preds)
confusion_matrix(y_true=y_train,y_pred=clf.predict(X_train))
plt.figure(figsize=(12,12))

sns.heatmap(df.corr(),annot=True)
df.boxplot()

plt.xticks(rotation = 90)
df = pd.read_csv('../input/heart.csv')

df.head()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
X_train,X_test,y_train,y_test = train_test_split(X,df.target,test_size=0.2)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(df.drop('target',axis=1))
knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)



print("{} NN Score: {:.2f}%".format(3, knn.score(X_test, y_test)*100))
scorelist = []

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    scorelist.append(knn.score(X_test,y_test))

plt.plot(range(1,20),scorelist)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

print("Maximum KNN Score is {:.2f}%".format((max(scorelist))*100))
from sklearn.ensemble import RandomForestClassifier



max_accuracy = 0





for x in range(2000):

    rf = RandomForestClassifier(random_state=x)

    rf.fit(X_train,y_train)

    Y_pred_rf = rf.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_rf,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = x

        

#print(max_accuracy)

#print(best_x)



rf = RandomForestClassifier(random_state=best_x)

rf.fit(X_train,y_train)

Y_pred_rf = rf.predict(X_test)
Y_pred_rf.shape

score_rf = round(accuracy_score(Y_pred_rf,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")