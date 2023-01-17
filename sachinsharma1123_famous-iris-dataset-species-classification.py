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
df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df
df.info()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(12,5))

sns.countplot(df['Species'])
#all thr three classes are balanced 
plt.figure(figsize=(12,5))

sns.scatterplot(x=df['SepalLengthCm'],hue=df['Species'],y=df['SepalWidthCm'],data=df)

#lets plot some barplot for more clear cut information
plt.figure(2,figsize=(12,5))

sns.barplot(x=df['Species'],y=df['SepalLengthCm'],data=df)
#it is clear from the above figure that setosa has the shorter sepal lenght and virginica has longer
#similarly we can plot for the sepal width

plt.figure(figsize=(12,5))

sns.barplot(x=df['Species'],y=df['SepalWidthCm'],data=df)
#here we can seee that setosa has the highest sepal with and versicolor has lower
plt.figure(figsize=(12,5))

sns.barplot(x=df['Species'],y=df['PetalLengthCm'],data=df)
#virginica corresponds to higher petal length and setosa has smaallest among all
plt.figure(figsize=(12,5))

sns.barplot(x=df['Species'],y=df['PetalWidthCm'],data=df)
# here also virginica corresponds to higher petal width and setosa has smaallest among all
#SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm

plt.figure(figsize=(12,5))

sns.lineplot(x=df['SepalLengthCm'],y=df['PetalLengthCm'],data=df)

# from the above figure we can conclude that both features tends to have linear relationship
plt.figure(figsize=(12,5))

sns.lineplot(x=df['SepalWidthCm'],y=df['PetalWidthCm'],data=df)

plt.figure(figsize=(12,5))

sns.heatmap(df.corr())
#lets delete the unnecessary columns from the dataset

df=df.drop(['Id'],axis=1)
#lets first convert the categorical data into integer target values

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Species']=le.fit_transform(df['Species'])
#lets separate the data into dependent and independent variables
#dependent variable

y=df['Species']
#independent variable

x=df.drop(['Species'],axis=1)
#split the data into train and test data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error



lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)

from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)
sns.scatterplot(x=list(range(1,21)),y=list_1)
#lets plot the decisoin boundary for the kneighbors classifier





from mlxtend.plotting import plot_decision_regions

from sklearn.decomposition import PCA

knn=KNeighborsClassifier(n_neighbors=3)

pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(x_train)

knn.fit(X_train2, y_train)

plt.figure(figsize=(12,5))

plot_decision_regions(X_train2, y_train.values, clf=knn, legend=2)



plt.xlabel(df.columns[0], size=14)

plt.ylabel(df.columns[1], size=14)

plt.title('k neighbors decision boundary', size=16)
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_2=svm.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2