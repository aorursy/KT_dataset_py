### IMPORTING LIBRARY

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets
df=pd.read_csv('../input/iris/Iris.csv')
df.head()
df.info()
df.describe()


def barchart(feature):

    setosa=df[df['Species']=='Iris-setosa'][feature].value_counts()

    virginica=df[df['Species']=='Iris-virginica'][feature].value_counts()

    versicolor=df[df['Species']=='Iris-versicolor'][feature].value_counts()

    #dead=df[df['Accident']==0][feature].value_counts()

    #survived1=survived[1]

    #dead1=dead[0]

    df1 = pd.DataFrame([setosa,virginica,versicolor])

    df1.index=['setosa','virginica','versicolor']

    df1.plot(kind='bar',stacked=True,figsize=(10,5))
barchart('SepalLengthCm')
df['SepalLengthCm'].value_counts().plot(kind='bar')
df['PetalLengthCm'].value_counts().plot(kind='bar')
df['PetalWidthCm'].value_counts().plot(kind='bar')
barchart('PetalLengthCm')
barchart('PetalWidthCm')
df['SepalLengthCm'].value_counts().plot(kind='pie')
df['PetalLengthCm'].value_counts().plot(kind='pie')
df['PetalWidthCm'].value_counts().plot(kind='pie')
## drop the useless and the target column
X= df.drop(['Id', 'Species'], axis=1)

Y = df['Species']
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
sample_result=[]

MachineLearningAlgo=[]

X=['LinearSVC','DecisionTreeClassifier','KNeighborsClassifier','SVC','GradientBoostingClassifier','RandomForestClassifier']

Z=[LinearSVC(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC(),GradientBoostingClassifier(),RandomForestClassifier()]
for model in Z:

    model.fit(X_train,y_train)      ## training the model this could take a little time

    accuracy=model.score(X_test,y_test)    ## comparing result with the test data set

    MachineLearningAlgo.append(accuracy) 

    sample_result.append(model.predict([[6, 3, 4, 2]]))

    ## saving the accuracy
d={'Accuracy':MachineLearningAlgo,'Algorithm':X}

df1=pd.DataFrame(d)
d1={'sample_result':sample_result,'Algorithm':X}

df2=pd.DataFrame(d1)
# testting with different types of nearest neighbour in KNN

from sklearn import metrics

klist = list(range(1,30))

scores = []

for k in klist:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))

    

plt.plot(klist, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()

df1
df2