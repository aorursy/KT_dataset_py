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
# importing alll the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
df=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

df.head()
df.shape
df.describe()
df.sample(5)
df.info()
# library & dataset

import seaborn as sns

#df = sns.load_dataset('iris')

 

# Make boxplot for one group only

sns.violinplot(x=df['species'],y=df["petal_width"])

#sns.plt.show()

sns.violinplot(x=df['species'],y=df["petal_length"])
sns.violinplot(x=df['species'],y=df["sepal_length"])
sns.violinplot(x=df['species'],y=df["sepal_width"])
# Make default histogram of sepal length

sns.distplot( df["petal_length"],bins=10)

#sns.plt.show()

 

# Control the number of bins

#sns.distplot( df["sepal_length"], bins=20 )

#sns.plt.show()

df
g=sns.pairplot(df,hue="species")
X = df.drop(['species'],axis=1).values

y = df['species'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print(y_pred)

knn.score(X_test,y_test)
metrics.accuracy_score(y_test,y_pred)
metrics.f1_score(y_test,y_pred,average='weighted')