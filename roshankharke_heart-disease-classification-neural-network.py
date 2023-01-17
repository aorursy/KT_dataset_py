# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_recall_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.svm import SVC

%matplotlib inline
df=pd.read_csv('../input/heart.csv')

df.head(5)
df.describe()
df.info()
plt.figure(figsize=(14,10))

sns.heatmap(df.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)

plt.show()
df.groupby('cp',as_index=False)['target'].mean()
df.groupby('slope',as_index=False)['target'].mean()
df.groupby('thal',as_index=False)['target'].mean()
df.groupby('target').mean()
sns.distplot(df['target'],rug=True)

plt.show()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(25,8),color=['gold','brown' ])

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,5),color=['cyan','coral' ])

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(df['target'], df['trestbps'], palette = 'rainbow')

plt.title('Relation of tresbps with target', fontsize = 10)
sns.pairplot(data=df)
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,5),color=['tomato','indigo' ])

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
plt.figure(figsize=(16,8))

plt.subplot(2,2,1)

plt.scatter(x=df.age[df.target==1],y=df.thalach[df.target==1],c='blue')

plt.scatter(x=df.age[df.target==0],y=df.thalach[df.target==0],c='black')

plt.xlabel('Age')

plt.ylabel('Max Heart Rate')

plt.legend(['Disease','No Disease'])



plt.subplot(2,2,2)

plt.scatter(x=df.age[df.target==1],y=df.chol[df.target==1],c='red')

plt.scatter(x=df.age[df.target==0],y=df.chol[df.target==0],c='green')

plt.xlabel('Age')

plt.ylabel('Cholesterol')

plt.legend(['Disease','No Disease'])



plt.subplot(2,2,3)

plt.scatter(x=df.age[df.target==1],y=df.trestbps[df.target==1],c='cyan')

plt.scatter(x=df.age[df.target==0],y=df.trestbps[df.target==0],c='fuchsia')

plt.xlabel('Age')

plt.ylabel('Resting Blood Pressure')

plt.legend(['Disease','No Disease'])



plt.subplot(2,2,4)

plt.scatter(x=df.age[df.target==1],y=df.oldpeak[df.target==1],c='grey')

plt.scatter(x=df.age[df.target==0],y=df.oldpeak[df.target==0],c='navy')

plt.xlabel('Age')

plt.ylabel('ST depression')

plt.legend(['Disease','No Disease'])

plt.show()
df.head(5)
X = df.drop(['target'], axis = 1)

y = df.target.values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

import keras

from keras.models import Sequential

from keras.layers import Dense

import warnings
classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 13))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
import seaborn as sns

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.round())

sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)

#accuracy score

from sklearn.metrics import accuracy_score

ac=accuracy_score(y_test, y_pred.round())

print('accuracy of the model: ',ac)
model_accuracy = pd.Series(data=[ac],index=['Neural Network'])

fig= plt.figure(figsize=(8,8))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')