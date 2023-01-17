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
df=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")
df.info() 

df_test.info()
df.describe()

df_test.describe()
df.isnull().sum()



df_test.isnull().sum()
(df.isnull().sum()/df.shape[0])*100
df=df.drop(['Cabin'],1)

df_test=df_test.drop(["Cabin"],1)
df.shape,df_test.shape
df['Embarked'].unique()
df['Embarked'].value_counts()
df['Embarked']=df['Embarked'].fillna('S')
df['Age']=df["Age"].fillna(df["Age"].mean())

df_test['Age']=df_test["Age"].fillna(df_test["Age"].mean())

df_test['Fare']=df_test["Fare"].fillna(df_test["Fare"].mean())
df.info(),df_test.info()
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  
df['Title'].head()

df_test['Title'].head()
pd.crosstab(df["Title"],df["Sex"])
df["Title"]=df["Title"].apply(lambda x:x if x in ["Mr","Mrs","Master","Miss"] else "Rare")

df_test["Title"]=df_test["Title"].apply(lambda x:x if x in ["Mr","Mrs","Master","Miss"] else "Rare")
df["Title"].describe()
df_test['Title'].describe()
df=df.drop(["Ticket"],1)

df_test=df_test.drop(["Ticket"],1)
df.shape,df_test.shape
df.info()
df=df.drop(["Name"],1)

df_test=df_test.drop(["Name"],1)
df_train = pd.get_dummies(df, prefix='Category_', columns=['Sex','Embarked','Title'])

df_test = pd.get_dummies(df_test, prefix='Category_', columns=['Sex','Embarked','Title'])
df_train.info()

df_test.info()
df_train.head()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

df_train["Age"]=pd.DataFrame(sc.fit_transform(pd.DataFrame(df_train["Age"])))

df_train["Fare"]=pd.DataFrame(sc.fit_transform(pd.DataFrame(df_train["Fare"])))

df_test["Age"]=pd.DataFrame(sc.fit_transform(pd.DataFrame(df_test["Age"])))

df_test["Fare"]=pd.DataFrame(sc.fit_transform(pd.DataFrame(df_test["Fare"])))
df_train.head()
Y= df_train["Survived"]

X= df_train.drop(columns = ["PassengerId","Survived"])

X_results = df_test.drop(columns=["PassengerId"])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns
model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)



# predict

y_pred = model_linear.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
model_kernel = SVC(kernel='rbf')

model_kernel.fit(X_train, y_train)



# predict

y_pred_ = model_kernel.predict(X_test)

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]





# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
model = SVC(C=100, gamma=0.01, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")
y_pred = model.predict(X_results)

y_pred=pd.DataFrame(y_pred)
y_pred.describe()
temp = pd.concat([df_test.PassengerId, y_pred], axis=1)

temp.columns = ['PassengerId', 'Survived']

temp.head()
temp.to_csv('submit_titanic.csv', index=False)