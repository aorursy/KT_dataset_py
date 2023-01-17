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
df = pd.read_csv(os.path.join(dirname, filename))

df
df['Type'].value_counts()
fig = plt.figure(figsize=(18,16))

df.boxplot(column=['RI','Mg','Na','Fe','Ba','Ca','K','Al','Si'])

fig.show()
df.corr()
sns.boxplot('Ca',data = df)
df['Ca'] = df['Ca'].rank()
sns.boxplot(df['Ca'].rank(),data = df)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
X = df.drop('Type',axis = 1)

y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

prediction_logreg = logreg.predict(X_test)

print(logreg.score(X_test,y_test))
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(prediction_logreg,y_test))

print(confusion_matrix(prediction_logreg,y_test))
rfmodel=RandomForestClassifier(n_estimators=100)

rfmodel.fit(X_train,y_train)

yp=rfmodel.predict(X_test)

print(rfmodel.score(X_test,y_test))

print(classification_report(yp,y_test))

print(confusion_matrix(yp,y_test))
error_rate = []

for i in range(1,40):

    count = 0

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    for i in range(0,len(pred_i)):

        if np.array(pred_i)[i]!=np.array(y_test)[1]:

            count+=1

    error_rate.append(count)
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Incorrect Prediction vs. K Value')

plt.xlabel('K')

plt.ylabel('Incorrect Prediction')
knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
knn.score(X_test,y_test)