# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import libraries:

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# import dataset:

df=pd.read_csv("../input/SAheart.csv")
df.head()
df.describe()
df.info()
# visualization :

sns.pairplot(df,hue='famhist')
sns.scatterplot(df['age'],df['sbp'],hue='famhist',data=df)
sns.scatterplot(df['age'],df['tobacco'],hue='famhist',data=df)
sns.scatterplot(df['age'],df['adiposity'],hue='famhist',data=df)
sns.scatterplot(df['age'],df['typea'],hue='famhist',data=df)
sns.scatterplot(df['age'],df['alcohol'],hue='famhist',data=df)
sns.scatterplot(df['age'],df['obesity'],hue='famhist',data=df)
sns.lineplot(df['age'],df['sbp'],hue='famhist',data=df)
sns.lineplot(df['age'],df['adiposity'],hue='famhist',data=df)
sns.lineplot(df['age'],df['tobacco'],hue='famhist',data=df)
sns.lineplot(df['age'],df['typea'],hue='famhist',data=df)
sns.lineplot(df['age'],df['obesity'],hue='famhist',data=df)
sns.lineplot(df['age'],df['alcohol'],hue='famhist',data=df)
sns.lmplot('age','sbp',hue='famhist',data=df)
sns.lmplot('age','adiposity',hue='famhist',data=df)
sns.lmplot('age','alcohol',hue='famhist',data=df)
sns.distplot(df['sbp'],bins=20,kde=False)
sns.distplot(df['obesity'],bins=20,kde=False)
sns.distplot(df['alcohol'],bins=20,kde=False)
sns.distplot(df['tobacco'],bins=20,kde=False)
sns.boxplot(df['sbp'],data=df)
sns.boxplot(df['alcohol'],data=df)
sns.boxplot(df['obesity'],data=df)
sns.boxplot(df['typea'],data=df)
sns.boxplot(df['tobacco'],data=df)
sns.boxplot(df['ldl'],data=df)
sns.boxplot(df['adiposity'],data=df)
X=df.iloc[:,:-1].values

y=df.iloc[:,9].values
X
y
# encode:

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

encode=LabelEncoder()

X[:,4]=encode.fit_transform(X[:,4])

encode=LabelEncoder()

df['chd']=encode.fit_transform(df['chd'])
#spliting the dataset into  training and test set:

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# feature sacling the dataset:

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
# fitting the dataseto SVM:

from sklearn.svm import SVC

classifier=SVC()

classifier.fit(X_train,y_train)
# prediction :

y_pred=classifier.predict(X_test)
y_pred
y_test
# making the confusion matrix:

from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,y_pred)
cm
(53+14)/93

# kernel-svm:

from sklearn.svm import SVC

classifier=SVC(kernel='linear',random_state=0)

classifier.fit(X_train,y_train)
# apply the k-flod cross validation :

from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies
accuracies.mean()
# now grid_search:

from sklearn.model_selection import GridSearchCV

parameters=[{'C':[1,10,100,1000],'kernel':['linear']},

           {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10

                        ,n_jobs=-1)

grid_search=grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_accuracy
best_parameters=grid_search.best_params_
best_parameters