#Importing the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import iqr



gl_df=pd.read_csv('../input/glass/glass.csv')



gl_df.head()
gl_df.info()
gl_df.describe()
col_names=gl_df.columns

col_names
#TO check yhe outliers using box-plot

for i in range(len(col_names)-1):

    plt.figure()

    sns.boxplot(x='Type',y=col_names[i],data=gl_df)
#To check the outliers using scatter plot

for i in range(len(col_names)-1):

    plt.figure()

    plt.scatter(gl_df['Type'],gl_df[col_names[i]])
#changing the dataframe to numpy array

data=gl_df.values
data
#Extracting the features and label data

X, y = data[:, :-1], data[:, -1]
X
y
#To remove the outliers

from sklearn.neighbors import LocalOutlierFactor

lof=LocalOutlierFactor()

yhat=lof.fit_predict(X)
#The values with -1 are outliers so we will remove them

yhat
mask=yhat!= -1

mask
#New feature and label data after removing outliers

X,y=X[mask,:],y[mask]

y
#Spliting dataset into train dataset and test dataset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21)
#fitting and predicting using KNN algorithm

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_predict=knn.predict(X_test)
#KNN Score

knn.score(X_test,y_test)
# Classification report and confusion metrics

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict,zero_division=0))
#Using Cross validation for k=3 and checking the value of n for knn for best result

from sklearn.model_selection import cross_val_score

neighbors=list(range(1,20))

cv_scores=[]

for n in neighbors:

    knn=KNeighborsClassifier(n_neighbors=n)

    score=cross_val_score(knn,X,y,cv=3,scoring='accuracy')

    cv_scores.append(score.mean())

    
#To determine the best value for n in KNN

import matplotlib.pyplot as plt

plt.plot(neighbors,cv_scores,marker='o')

plt.xlabel("neighbors")

plt.ylabel("accuracy_score")

plt.xticks(np.arange(1,21),neighbors)

plt.grid()

plt.show()
print(f"The score at n=12 is {cv_scores[11]}.")