import numpy as np

import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/hospital-readmissions/train.csv")

data.head()
sns.heatmap( data.corr() )
corrMatrix = data.corr()

colsForDelete = []

neededCols = list(data.columns)



for i in corrMatrix.columns:

    for j in corrMatrix.index:

        if(i!=j):

            if( abs(corrMatrix[i][j])>0.05 ):

                colsForDelete.append(i)



colsForDelete = list( pd.Series(colsForDelete).unique() )

print(len(colsForDelete))
data = data.drop( labels=colsForDelete, axis=1 )

data.head()
sns.heatmap( data.corr() )
for column in list(data.columns):

    print(column)
from sklearn.tree import DecisionTreeClassifier
femaleClassifier = DecisionTreeClassifier(

    min_samples_leaf = 100

)
X = data.drop( labels="gender_Female", axis=1 )

y = data["gender_Female"]



sns.heatmap( X.corr() )
femaleClassifier.fit(X, y)

print( "R^2="+str(femaleClassifier.score(X, y)) )