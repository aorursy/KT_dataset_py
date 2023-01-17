# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df
df['gender']=df['gender'].apply(lambda x: 1 if x=='female' else 0)
df['gender'].value_counts()
df['gender']=df['gender'].astype('float')
def plot_cat(cat_var):
    sns.barplot(x=cat_var,y='gender',data=df)
    plt.show()
plot_cat('race/ethnicity')
plt.figure(figsize=(13,5))
plot_cat('parental level of education')
plot_cat('lunch')
plot_cat('test preparation course')
sns.barplot(x='reading score',y='lunch',data=df,hue='gender')
sns.barplot(x='math score',y='lunch',data=df,hue='gender')
sns.barplot(x='writing score',y='lunch',data=df,hue='gender')

col=['math score','reading score','writing score']
af=df.loc[:,col]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
af=pd.DataFrame(sc.fit_transform(af),columns=col)
df=df.drop(col,axis=1)
df=pd.concat([df,af],axis=1)
df.head()
dff=pd.get_dummies(df)
X=dff.iloc[:,dff.columns!='gender']
y=dff.iloc[:,dff.columns=='gender']
dff.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train.values.ravel())
y_pred=classifier.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train.values.ravel(),cv=10)
print(accuracies)
print(accuracies.mean())

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
rfe=RFE(classifier,5)
rfe.fit(X_train,y_train.values.ravel())
X_train.columns[rfe.support_]
y_pred=rfe.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train.values.ravel(),cv=10)
print(accuracies)
print(accuracies.mean())

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=1)
X_train=lda.fit_transform(X_train,y_train.values.ravel())
X_test=lda.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train.values.ravel())
y_pred=classifier.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train.values.ravel(),cv=10)
print(accuracies)
print(accuracies.mean())
#
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid = GridSearchCV(SVC(),param_grid,cv=10)
grid.fit(X_train,y_train.values.ravel())
grid.best_params_,grid.best_estimator_
y_pred=grid.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='g')

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train.values.ravel(),cv=10)
print(accuracies)
print(accuracies.mean())
print('Accuracy: Logistic Regression:89.87%')
print(' Logistic Regression with Feature Extraction(5 features):89.87%')
print (' Logistic Regression with LDA:90.75%')
print('Grid Search with SVC:90.5')
      
