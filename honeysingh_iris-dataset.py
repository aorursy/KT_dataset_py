# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#input the Iris and drop the column 'Id'
df = pd.read_csv('../input/Iris.csv')
df.drop(['Id'],axis=1,inplace=True)
df.head()
#Verify that there are no nulls
df.isnull().sum()
#Just check the output frequency
#If they are unequal, care should be done to mix it in train and test set(stratify)
import seaborn as sns
sns.countplot(x='Species',data=df)
plt.show()
df.columns
columns = df.columns[:-1]
length = len(columns)
for i,j in zip(columns,range(length)):
    plt.subplot(length,3,j+1)
    df[i].hist(bins=20)
    plt.title(i)
plt.show
sns.pairplot(data=df,hue='Species',diag_kind='kde')
plt.show()
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

Y = df['Species']
X= df.drop(['Species'],axis=1)
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=.2)
from sklearn import metrics
def results(train_X,train_Y,test_X,test_Y):
    abc=[]
    classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']
    models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]
    for i in models:
        model = i
        model.fit(train_X,train_Y)
        prediction=model.predict(test_X)
        abc.append(metrics.accuracy_score(prediction,test_Y))
    models_dataframe=pd.DataFrame(abc,index=classifiers)   
    models_dataframe.columns=['Accuracy']
    print(models_dataframe)
results(train_X,train_Y,test_X,test_Y)
#Try to increase the results using feature Engineering
#1.correlation matrix
sns.heatmap(df.drop(['Species'],axis=1).corr(),annot=True,cmap='RdYlGn')
#Used to get the ranking of features
from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
from sklearn.preprocessing import StandardScaler #Standardisation
X=StandardScaler().fit_transform(X)
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=.2)
results(train_X,train_Y,test_X,test_Y)
#Ensemble is used  to combine multiple models with weightage to comeup with better results
linear_svc=svm.SVC(kernel='linear',C=0.1,gamma=10,probability=True)
radial_svm=svm.SVC(kernel='rbf',C=0.1,gamma=10,probability=True)
lr=LogisticRegression(C=0.1)

from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Radial_svm', radial_svm)], 
                       voting='soft', weights=[2,1]).fit(train_X,train_Y)
print('The accuracy for Linear and Radial SVM is:',ensemble_lin_rbf.score(test_X,test_Y))
ensemble_rad_lr_lin=VotingClassifier(estimators=[('Radial_svm', radial_svm), ('Logistic Regression', lr),('Linear_svm',linear_svc)], 
                       voting='soft', weights=[2,1,3]).fit(train_X,train_Y)
print('The ensembled model with all the 3 classifiers is:',ensemble_rad_lr_lin.score(test_X,test_Y))

