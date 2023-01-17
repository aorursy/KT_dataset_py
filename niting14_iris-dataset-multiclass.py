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
# Let's read the file first

# Always use the error handiling function while read the file



fname = "../input/iriscsv/iris.csv"

try:

    df_iris = pd.read_csv(fname)

    print("File has been loaded succesfully.")

except:

    print("Could not open the file.", fname)
df_iris.head(5) # Display the first 5 observations
df_iris.tail(5) # Display the last 5 observations
df_iris.shape # It will show the shape of the datatset (rows, cols)
# let's check the null value and duplicate values, if any. 

print("There are",df_iris.isnull().sum().sum(),"null values present in the datatset.")

print("There are",df_iris.duplicated().sum(),"duplicate obsrvations present in the datatset.")
# Better to remove the duplicated values before making the model.

if df_iris.duplicated().sum() > 0 :

    df_iris.drop_duplicates(inplace = True)

# Let's check the shape of the dataset again.

df_iris.shape
# Let's see the five point summpary of the dataset, which is neccessary to know about your dataset.

df_iris.describe().T

# Count = number of observations

# Mean = Average of all 149 observations for the feature

# STD = Standard deviation of the feature

# MIN = Minimum value between all 149 obervations

# 25% = value of the 25% positions

# 50% = value of the 50% positions (Median)

# 75% = value of the 75% positions

# Max = Maximum value between all 149 observations
# Libraries to plot some graph. 

import matplotlib.pyplot as plt

import seaborn as sns
num = 1

plt.figure(figsize = (20,3))

for fea in df_iris.drop("species", axis = 1).columns:

    plt.subplot(1,4,num)

    ax = sns.distplot(df_iris[fea])

    ax = plt.title(fea + " Dist Plot")

    num +=1

plt.subplots_adjust(wspace = 0.5)

plt.show()
num = 1

plt.figure(figsize = (20,4))

for fea in (df_iris.drop("species", axis = 1).columns):

    plt.subplot(1,4,num)

    ax=sns.boxplot(y=fea, data=df_iris)

    ax = plt.title(fea + " Box Plot")

    num +=1

plt.subplots_adjust(wspace = 0.5)

plt.show()
sns.pairplot(df_iris,diag_kind='kde')
ax = sns.heatmap(df_iris.drop("species", axis = 1).corr(), annot= True)

b, t = ax.get_ylim()

ax.set_ylim(b + 0.5, t - 0.5)

plt.show()
df_iris['species'].value_counts()
df_iris.replace({'versicolor' : 1, 'setosa' : 2 , 'virginica' : 3}, inplace = True)
X = df_iris.drop('species', axis=1)

Y = df_iris[['species']]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X, Y, test_size=.20, random_state = 17)

print("X_train Shape : ", X_train.shape)

print("y_train Shape : ", y_train.shape)

print("X_test Shape : ", X_test.shape)

print("y_test Shape : ", y_test.shape)
X_test.value_counts()

y_test.value_counts()
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

import warnings 

warnings.filterwarnings('ignore')
lrmodel = LogisticRegression(solver='liblinear', multi_class='ovr', random_state = 9 )

lrmodel.fit(X_train,y_train)

print("Train dataset accuracy score : ",lrmodel.score(X_train,y_train))

print("Test  dataset accuracy score : ",lrmodel.score(X_test,y_test))
lrmodel = LogisticRegression(solver='liblinear', multi_class='ovr', random_state = 9 )

lrmodel.fit(X_train,y_train)

print("Train dataset accuracy score : ",lrmodel.score(X_train,y_train))

print("Test  dataset accuracy score : ",lrmodel.score(X_test,y_test))
train_predict = lrmodel.predict(X_train)

test_predict = lrmodel.predict(X_test)

print("Confusion matrix of Train dataset: \n",confusion_matrix(y_train, train_predict))

print("Confusion matrix of Test dataset: \n",confusion_matrix(y_test, test_predict))
print("Classification report of Train dataset: \n", classification_report(y_train, train_predict))

print("Classification report of test dataset: \n", classification_report(y_test, test_predict))
dtree = DecisionTreeClassifier(random_state = 9)

dtree.fit(X_train,y_train)

print("Train dataset accuracy score : ",dtree.score(X_train,y_train))

print("Test  dataset accuracy score : ",dtree.score(X_test,y_test))