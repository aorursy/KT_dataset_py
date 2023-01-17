# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/pulsar_stars.csv')
# look first five data 

df.head()
df.info()
# summary statistic data 

df.describe()
df.target_class.value_counts()

sns.countplot(df.target_class)

plt.show()
# pair plot 

sns.pairplot(data=df,

             diag_kind="kde", 

             markers=".",

             plot_kws=dict(s=50, edgecolor="b", linewidth=1),

             hue="target_class",

             vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"],

                   diag_kws=dict(shade=True))



plt.tight_layout()

plt.show() 
# heatmap 

df.corr()

sns.heatmap(df.corr(),cmap="YlGnBu",linewidths=.5)

plt.title('Corelation Heatmap')

plt.show()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, cmap="YlGnBu",fmt= '.1f',ax=ax)

plt.title('Corelation Map')

plt.show()
# violinplot

# data normalization

df_nor=(df - np.min(df))/(np.max(df)-np.min(df))



sns.violinplot(data=df,y=" Mean of the integrated profile",x="target_class")

plt.show()



sns.violinplot(data=df,y=" Mean of the DM-SNR curve",x="target_class")

plt.show()
#Set x and y values

y=df.target_class.values

x_df=df.drop(['target_class'],axis=1)

#normalization

x=(x_df-np.min(x_df))/(np.max(x_df)-np.min(x_df))
# train/test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

print('lr accuracy :', lr.score(x_test,y_test))



# confusion matrix

y_pred = lr.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_true,y_pred)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

print('knn accuracy :',knn.score(x_test,y_test))

# confisioun matrix

y_pred = knn.predict(x_test)

y_true = y_test



# confisuon matrix

from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_true,y_pred)

score_list=[]

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,15),score_list)

plt.xlabel('k values')

plt.ylabel('accuracy')

plt.show()
from sklearn.svm import SVC

svm=SVC(random_state=1)

svm.fit(x_train,y_train)

print('svm accuracy :', svm.score(x_test,y_test))



# confisuon matrix

y_pred = svm.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix



cm_svm = confusion_matrix(y_true,y_pred)
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

print('nb accuracy : ', nb.score(x_test,y_test))



# confisuon matrix

y_pred = nb.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix



cm_nb = confusion_matrix(y_true,y_pred)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

print('dt.accuracy : ', nb.score(x_test,y_test))



# confisuon matrix

y_pred = dt.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix



cm_dt = confusion_matrix(y_true,y_pred)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(x_train,y_train)

print('rf accuracy : ', rf.score(x_test,y_test))



# confision matrix

y_pred = rf.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix



cm_rf = confusion_matrix(y_true,y_pred)
plt.figure(figsize=(20,15))



plt.suptitle("Confusion Matrixes",fontsize=20)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,cbar=False,annot=True,cmap="Greens",fmt="d")



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,cbar=False,annot=True,cmap="Greens",fmt="d")



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,cbar=False,annot=True,cmap="Greens",fmt="d")



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,cbar=False,annot=True,cmap="Greens",fmt="d")



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dt,cbar=False,annot=True,cmap="Greens",fmt="d")



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,cbar=False,annot=True,cmap="Greens",fmt="d")



plt.show()