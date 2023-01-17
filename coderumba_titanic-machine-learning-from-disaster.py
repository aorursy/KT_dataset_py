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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
pd.set_option("Display.max_rows", None)
training_data=pd.read_csv("/kaggle/input/titanic/train.csv")

test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

training_data.head()
missing_values_train=training_data.isnull().sum().sort_values(ascending=False)

missing_percent_train=(missing_values_train/training_data.isnull().count())

missing_train=pd.concat([missing_values_train, missing_percent_train], axis=1,keys=['Total','Percent'])

print(missing_train)
training_data.drop('Cabin', axis=1, inplace=True)

training_data.Age.fillna(training_data.Age.median(), inplace=True)

training_data.dropna(inplace=True)
training_data.dropna().head(3)
check_missing=training_data.isnull().sum().to_frame()

print(f"Missing value {check_missing}")
training_data.drop(['PassengerId'], axis=1, inplace=True)

training_data.head(3)
training_data['numeric_ticket'] = training_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
training_data['numeric_ticket'].value_counts()
training_data.head(3)
training_data['Fare']=training_data['Fare']/training_data['Fare'].max()

training_data['Age']=training_data['Age']/training_data['Age'].max()

training_data.head()
test_data.isnull().sum()
#Drop Cabin variable



test_data.drop(['Cabin'],axis=1, inplace=True)

test_data.head()
#Replace the missing Age values with its median value



test_data.fillna(test_data.Age.median(), axis=1, inplace=True)
#Check if we still have the missing values



test_data.isnull().sum()
#Divide Ticket variable into numeric and non-numeric variables



test_data['numeric_ticket'] = test_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
test_data.head(3)
#Simple feature scaling



test_data['Fare']=test_data['Fare']/test_data['Fare'].max()

test_data['Age']=test_data['Age']/test_data['Age'].max()

test_data.head()
training_data.dtypes.to_frame()
train_num=training_data[['Age','Fare']]

train_categ=training_data[['Survived','Pclass','Sex','numeric_ticket', 'Embarked','SibSp','Parch']]
f, ax=plt.subplots(figsize=(12,6))

sns.distplot(train_num.Age)
plt.subplots(figsize=(12,6))

sns.distplot(train_num.Fare)
#Let's if how normal distribution of Fare variable look like



fare_norm=np.log1p(train_num.Fare)

plt.subplots(figsize=(12,6))

sns.distplot(fare_norm)
training_data.drop(['Name'], axis=1, inplace=True)

training_data.head()
for category in train_categ.columns:

    plt.subplots(figsize=(10,5))

    sns.barplot(train_categ[category].value_counts().index, train_categ[category].value_counts()).set_title(category)

    plt.show()
training_data.groupby(['Sex'])['Survived'].value_counts().to_frame()
training_data.groupby(['Age'])['Survived'].value_counts().to_frame()
training_data.groupby(['SibSp'])['Survived'].value_counts().to_frame()
training_data.groupby(['Embarked'])['Survived'].value_counts().to_frame()
training_data.groupby(['numeric_ticket'])['Survived'].value_counts().to_frame()
bins = np.linspace(training_data.Fare.min(), training_data.Fare.max(), 10)

g = sns.FacetGrid(training_data, col="Survived", hue="Sex", palette="Set1", col_wrap=2, height=5)

g.map(plt.hist, 'Fare', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(training_data.Age.min(), training_data.Age.max(), 5)

g = sns.FacetGrid(training_data, col="Survived", hue="Sex", palette="Set1", col_wrap=2, height=5)

g.map(plt.hist, 'Age', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(training_data.SibSp.min(), training_data.SibSp.max(), 5)

g = sns.FacetGrid(training_data, col="Survived", hue="Sex", palette="Set1", col_wrap=2, height=5)

g.map(plt.hist, 'SibSp', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(training_data.Parch.min(), training_data.Parch.max(), 10)

g = sns.FacetGrid(training_data, col="Survived", hue="Sex", palette="Set1", col_wrap=2, height=5)

g.map(plt.hist, 'Parch', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(training_data.Age.min(), training_data.Age.max(), 10)

g = sns.FacetGrid(training_data, col="Survived", hue="Embarked", palette="Set1", col_wrap=2, height=5)

g.map(plt.hist, 'Age', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins=np.linspace(training_data.Age.min(), training_data.Age.max(),10)

g=sns.FacetGrid(training_data, col='Survived', hue='Pclass', palette='Set1', col_wrap=2, height=5)

g.map(plt.hist, 'Age', bins=bins, ec='k')

g.axes[-1].legend()

plt.show()
g=sns.FacetGrid(training_data, col="Survived",  hue="Sex", height=7)

g.map(plt.scatter, "Age", "Fare" , edgecolor="w").add_legend()
table1=pd.pivot_table(training_data,columns=['Survived'], values='Fare',index=['Pclass', 'Sex'], aggfunc="count")

table1
table3=pd.pivot_table(training_data, values='Fare', columns=['Survived'], index=['Embarked','Sex'], aggfunc="count")

table3
training_data.describe()
training_data.corr()
plt.subplots(figsize=(10,7))

sns.heatmap(training_data.corr(), cmap='magma',cbar=True,center=0, linewidth=1)
traindata_titanic=training_data
traindata_titanic['Fare']=np.log1p(traindata_titanic.Fare)
dummies_df=pd.get_dummies(traindata_titanic, columns=['Pclass','Sex','SibSp','Parch','Embarked','numeric_ticket'])

dummies_df.head()
len(dummies_df.columns)
df_train=dummies_df.drop(['Survived','Ticket'], axis=1)

df_train.head(3)
from sklearn.model_selection import train_test_split

X=df_train

y=traindata_titanic['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=1)

print(f"Number of test samples: {X_test.shape[0]}")

print(f"Number of train samples: {X_train.shape[0]}")
import warnings

warnings.filterwarnings('ignore')
#Decision Tree



from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4, random_state=1)

dt.fit(X_train,y_train)
dt.score(X_train,y_train)
from sklearn.model_selection import cross_val_score

cv_dt= cross_val_score(dt,X_train,y_train,cv=10)

print(cv_dt)

print(cv_dt.mean())
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
knn.score(X_train, y_train)
#Cross validation score

cv_knn= cross_val_score(knn,X_train,y_train,cv=10)

print(cv_knn)

print(cv_knn.mean())
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 2000)

lr.fit(X_train, y_train)
lr.score(X_train, y_train)
#Cross validation score

cv_lr= cross_val_score(lr,X_train,y_train,cv=10)

print(cv_lr)

print(cv_lr.mean())
from sklearn.model_selection import GridSearchCV 



lr = LogisticRegression()

param_grid = {'max_iter' : [2000],

              'penalty' : ['l1', 'l2'],

              'C' : np.logspace(-4, 4, 20),

              'solver' : ['liblinear']}



clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)

best_clf_lr = clf_lr.fit(X_train,y_train)

best_clf_lr
#Parameter setting that gave the best results on the hold out data

best_parameters_lr=best_clf_lr.best_params_

print(best_parameters_lr)
#Best score

best_result_lr=best_clf_lr.best_score_

print(best_result_lr)
knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : [3,5,7,9],

              'weights' : ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree','kd_tree'],

              'p' : [1,2]}

clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)

best_clf_knn = clf_knn.fit(X_train,y_train)

best_clf_knn
#Parameter setting that gave the best results on the hold out data

best_parameters_knn=best_clf_knn.best_params_

print(best_parameters_knn)
#Best score

best_result_knn=best_clf_knn.best_score_

print(best_result_knn)
dt=DecisionTreeClassifier(random_state=1)

param_grid=[{'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_depth':[15, 20, 25],

             'max_features':['auto', 'sqrt', 'log2'],'min_samples_leaf': [2,3],'min_samples_split': [2,3]}]

clf_dt = GridSearchCV(dt, param_grid = param_grid, cv = 10, verbose = True, n_jobs = -1)

best_clf_dt = clf_dt.fit(X_train,y_train)

best_clf_dt
#Parameter setting that gave the best results on the hold out data

best_parameters_dt=best_clf_dt.best_params_

print(best_parameters_dt)
#Best score

best_result_dt=best_clf_dt.best_score_

print(best_result_dt)
test_data.head(3)
test_data.shape
df_test=pd.get_dummies(test_data, columns=['Pclass','Sex','SibSp','Parch','Embarked','numeric_ticket'])

df_test.head(3)
wantedcols=X_train.columns

wantedcols
predictions=best_clf_knn.predict(df_test[wantedcols])

predictions[:20]
submission=pd.DataFrame()

submission['PassengerId']=df_test['PassengerId']

submission['Survived']=predictions

submission.head(10)
#Are our test and submission dataframe have same length?



if len(submission)==len(test_data):

    print(f"Submission dataframe is the same length as test rows: {len(submission)}")

else:

    print("Dataframe mismatched.")
#Save submission dataframe to csv for sumbission for kaggle competition

submission.to_csv("rumba_submission_.csv", index=False)