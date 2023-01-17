# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/train.csv')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df.head()
df.info()
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')
df['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age) :

        

        if Pclass == 1 :

            return 37

        if Pclass == 2 :

            return 29

        else :

            return 24

        

    else : 

        return Age

    

df['Age'] = df[['Age','Pclass']].apply(impute_age, axis =1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop('Cabin',axis =1, inplace = True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.info()
sex = pd.get_dummies(df['Sex'],drop_first=True)

embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df = pd.concat([df,sex,embark],axis=1)
df.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), 

                                                    df['Survived'], test_size=0.30, 

                                                    random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)



from sklearn import metrics



from sklearn.metrics import confusion_matrix, classification_report



print('The accuracy of the Logistic Regression is',metrics.accuracy_score(predictions, y_test))
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test, predictions))
from sklearn.svm import SVC
SVCmodel = SVC()

SVCmodel.fit(X_train, y_train)
predictionsSVC = SVCmodel.predict(X_test)

print('The accuracy of the Support Vector Machine is',metrics.accuracy_score(predictionsSVC, y_test))
print(classification_report(y_test,predictionsSVC))

print(confusion_matrix(y_test, predictionsSVC))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid_predictions = grid.predict(X_test)

print('The accuracy of the Support Vector Machine with Grid Search is',metrics.accuracy_score(grid_predictions, y_test))
print(classification_report(y_test,grid_predictions))

print(confusion_matrix(y_test, grid_predictions))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train,y_train)
KNNpredictions = knn.predict(X_test)
print('The accuracy of KNN with 1 neighbor is',metrics.accuracy_score(KNNpredictions, y_test))
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn2 = KNeighborsClassifier(n_neighbors = 7)

knn2.fit(X_train,y_train)

KNN2predictions = knn2.predict(X_test)
print('The accuracy of KNN with 7 neighbors is',metrics.accuracy_score(KNN2predictions, y_test))