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
# Importing necessary libraries

import os  

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

%matplotlib inline
# Loading the dataset in python with the name df and displaying first 20 rows: 

df=pd.read_csv("/kaggle/input/wine-quality/winequalityN.csv")

df.head(20)
#Getting an overall picture of the data types and shape of our dataset :

df.info()
# Taking sum of all missing values in each column:

df.isna().sum()
df=df.fillna(df.mean())
df.describe()
import seaborn as sns

sns.set(rc={'figure.figsize':(10,8)})

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.show()
sns.boxplot(x="type",y="quality",data=df, palette="dark")

plt.show()
g = sns.PairGrid(df, y_vars=["quality"], x_vars=list(df)[1:-6],palette="GnBu_d",hue="type")

g.map(sns.regplot)

g.set(ylim=(-1, 11), yticks=[0, 5, 10]);

g.add_legend()

plt.show()



a = sns.PairGrid(df, y_vars=["quality"], x_vars=list(df)[-6:-1],palette="GnBu_d",hue="type")

a.map(sns.regplot)

a.set(ylim=(-1, 11), yticks=[0, 5, 10]);

a.add_legend()

plt.show()
df=df[df.columns.drop('type')]
x=df[df.columns.drop("quality")]

normalized_x=preprocessing.minmax_scale(x)

y=df["quality"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression

clf= LinearRegression().fit(X_train,y_train)

clf.score(X_train,y_train)
from sklearn.metrics import mean_squared_error

a=clf.predict(X_train)

train_rmse= (mean_squared_error(a,y_train)) ** 0.5

print(train_rmse)

b=clf.predict(X_test)

test_rmse= (mean_squared_error(b,y_test)) ** 0.5

test_rmse
from sklearn.linear_model import RidgeCV

alpha= np.arange(0.01,10,0.1).tolist()

clf = RidgeCV(alphas=alpha).fit(X_train, y_train)

score=clf.score(X_train, y_train)

print("R^2 =", score)

a=clf.predict(X_train)

train_rmse= (mean_squared_error(a,y_train)) ** 0.5

print("train_rmse = ", train_rmse)

b=clf.predict(X_test)

test_rmse= (mean_squared_error(b,y_test)) ** 0.5

print("test_rmse = ", test_rmse)

from sklearn.linear_model import LassoCV

alpha= np.arange(0.01,10,0.1).tolist()

clf = LassoCV(alphas=alpha).fit(X_train, y_train)

score=clf.score(X_train, y_train)

print("R^2 =", score)

a=clf.predict(X_train)

train_rmse= (mean_squared_error(a,y_train)) ** 0.5

print("train_rmse = ", train_rmse)

b=clf.predict(X_test)

test_rmse= (mean_squared_error(b,y_test)) ** 0.5

print("test_rmse = ", test_rmse)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

knn=KNeighborsClassifier(n_neighbors=80)

knn.fit(X_train,y_train)

g=knn.predict(X_test)

metrics.accuracy_score(y_test,g)

metrics.f1_score(y_test,g,average="micro")
from sklearn import svm

SV=svm.SVC(C=1,kernel='rbf')

SV.fit(X_train,y_train)

g=SV.predict(X_test)

metrics.accuracy_score(y_test,g)

metrics.f1_score(y_test,g,average="micro")
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(X_train,y_train)

g=nb.predict(X_test)

metrics.accuracy_score(y_test,g)

metrics.f1_score(y_test,g,average="micro")
from sklearn.linear_model import LogisticRegression

f=LogisticRegression(max_iter=10000)

f.fit(X_train,y_train)

g=f.predict(X_test)

metrics.accuracy_score(y_test,g)

metrics.f1_score(y_test,g,average="micro")
from sklearn.tree import DecisionTreeClassifier

f= DecisionTreeClassifier()

f.fit(X_train,y_train)

g=f.predict(X_test)

print(metrics.accuracy_score(y_test,g))

print(metrics.f1_score(y_test,g,average="micro"))
from sklearn.ensemble import RandomForestClassifier

f= RandomForestClassifier()

f.fit(X_train, y_train)

g=f.predict(X_test)

print(metrics.accuracy_score(y_test,g))
def qual(a):

  if 1<=a["quality"]<=3 :

    return "bad wine"

  elif 4<=a["quality"]<=7 :

    return "good wine"

  else :

    return "excellent wine"

df["wine_qual"]=df.apply(qual,axis=1)

df.head()



y=df["wine_qual"]

X_train, X_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.33, random_state=42)



from sklearn.ensemble import RandomForestClassifier

f= RandomForestClassifier()

f.fit(X_train, y_train)

g=f.predict(X_test)

print("accuracy for RandomForestClassifier:", metrics.accuracy_score(y_test,g))
from sklearn import svm

SV=svm.SVC(C=1,kernel='rbf')

SV.fit(X_train,y_train)

g=SV.predict(X_test)

metrics.accuracy_score(y_test,g)

print("accuracy for Support Vector Machines: ", metrics.f1_score(y_test,g,average="micro"))