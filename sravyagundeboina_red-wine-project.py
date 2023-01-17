#Import all necessary modules



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix 

from sklearn import metrics
#Importing the data set



df=pd.read_csv("/kaggle/input/red-wine-quality/red_wine_quality.csv")

df.head(10)
#check for null values



df.info()
#gives number of null values in each column.



df.isnull().sum()
#drawing histograms



plt.hist(df['chlorides'],bins=10,color='lightblue',ec='black')

plt.xlabel("chlorides")

plt.ylabel("Frequency")

plt.show()



plt.hist(df['density'],bins=10,color='lightblue',ec='black')

plt.xlabel("density")

plt.ylabel("Frequency")

plt.show()



plt.hist(df['pH'],bins=10,color='lightblue',ec='black')

plt.xlabel("pH")

plt.ylabel("Frequency")

plt.show()



plt.hist(df['sulphates'],bins=10,color='lightblue',ec='black')

plt.xlabel("sulphates")

plt.ylabel("Frequency")

plt.show()



plt.hist(df['alcohol'],bins=10,color='lightblue',ec='black')

plt.xlabel("alcohol")

plt.ylabel("Frequency")

plt.show()



plt.hist(df['citric acid'],color='lightblue',ec='black')

plt.xlabel("citric acid")

plt.ylabel("Frequency")

plt.show()
#box plots for all numerical columns



for cl in df:

    plt.figure()

    plt.ylabel("frequency")

    df.boxplot([cl],color='red')
#scatter plots



plt.scatter(x=df['fixed acidity'],y=df['alcohol'],marker='D',color='black')

plt.xlabel("fixed acidity")

plt.ylabel("alcohol")

plt.show()



plt.scatter(x=df['volatile acidity'],y=df['sulphates'],marker='v',color='violet')

plt.xlabel("volatile acidity")

plt.ylabel("sulphates")

plt.show()



plt.scatter(x=df['citric acid'],y=df['pH'],marker='*',color='green')

plt.xlabel("citric acid")

plt.ylabel("pH")

plt.show()



plt.scatter(x=df['residual sugar'],y=df['density'],marker='X',color='red')

plt.xlabel("residual sugar")

plt.ylabel("density")

plt.show()



plt.scatter(x=df['chlorides'],y=df['total sulfur dioxide'],marker='p',color='blue')

plt.xlabel("chlorides")

plt.ylabel("total SO2")

plt.show()
#splitting the data into train and test

#fixed acidity, free sulfur dioxide, total sulfur dioxide, sulphates, alcohol are trained



train=["fixed acidity", "free sulfur dioxide", "total sulfur dioxide", "sulphates", "alcohol"]

X = df[train]

y = df["quality"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=15,stratify=y)

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)
#accuracy score



acc=knn.score(X_test,y_test)

print('percentage accuracy :{:f} %'.format(100*acc))