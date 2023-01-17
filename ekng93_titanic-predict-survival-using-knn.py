# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Libraries

from matplotlib import pyplot as plt

import pandas as pd



data_url = '../input/titanic/train.csv'



df = pd.read_csv(data_url, index_col=0) #index_col=0 means first column will become index, otherwise specify with column name 'example name'





print (df.head())

print (df.dtypes)

print (df.shape)

print (df.columns)
df=pd.get_dummies(df, columns=['Sex'])

df.head()
corr = df.corr()

print(corr)



import statsmodels.api as sm

sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.show()
df_2 = df [['Pclass','Fare','Survived','Sex_female','Sex_male']]

df_2.isna().sum()
X = df_2[['Pclass','Fare','Sex_female','Sex_male']].values

y = df_2['Survived'].values
from sklearn.model_selection import KFold

kf = KFold(n_splits=7) 

kf.get_n_splits(X)

print(kf)

print (kf.split(X))



for train_index, test_index in kf.split(X):

 print('TRAIN:', train_index, 'TEST:', test_index)

 X_train, X_test = X[train_index], X[test_index]

 y_train, y_test = y[train_index], y[test_index]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, X_train, y_train)

print("Cross-validation scores: {}".format(scores))
y_predict=knn.predict(X_test)



#print(len(X_test))

#print(y_test[1])

#fig = plt.figure()

#ax = fig.add_subplot(111) #a new axes is created at the first position (1) on a grid of 2 rows and 3 columns

#ax = fig.add_subplot(233)

#ax = fig.add_axes([2,2,2,2]) 

#ax = fig.add_axes([1,1,2,2]) #[x0, y0, width, height], x0 and y0 = lower left point of graph

#i=0

#for i in range (0,len(X_test)-1):

#    ax.bar(i + 0.00, y_test[i], color = 'b', width = 0.25)

#    ax.bar(i + 0.25, y_predict[i], color = 'g', width = 0.25)

#plt.show()



#tabulate = pd.DataFrame({'predict':y_predict, 'actual':y_test})

#temp2=tabulate.head(21)

#print (temp2)

#temp2.plot(kind='bar',figsize=(10,8))

#plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

#plt.show()
count=0

for i in range (0,len(y_test)):

    if y_test[i] == y_predict[i]:

        count=count+1

    else:

        pass



accuracy = (count/len(y_predict))*100

print ("Model accuracy = ", accuracy)

#Libraries

from matplotlib import pyplot as plt

import pandas as pd



data_url = '../input/titanic/test.csv'



df = pd.read_csv(data_url, index_col=0) #index_col=0 means first column will become index, otherwise specify with column name 'example name'



ind = df.index

print (df.head())

print (df.dtypes)

print (df.shape)

print (df.columns)

print (df.index)
df.describe()
df.isna().sum()
median = df['Fare'].median()

df['Fare'].fillna(median, inplace=True)
df.isna().sum()
df=pd.get_dummies(df, columns=['Sex'])

df.head()
X = df[['Pclass','Fare','Sex_female','Sex_male']].values

y_predict = knn.predict(X)
df_save = pd.DataFrame({'PassengerId':ind,'Survived':y_predict})

df_save.to_csv("predict1.csv") 