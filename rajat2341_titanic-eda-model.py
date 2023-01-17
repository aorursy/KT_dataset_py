# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load in our Libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read Data

df = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

df.head()
test_original = test.copy()
test.head()
# Shape of the dataset.

df.shape, test.shape
# Checking null values.

df.info()

print('*'*50)

test.info()
# Checking Quantiles.

df.describe()
# Describing categorical values.

df.describe(include = ['object'])
# Print Data types for each variable.

df.dtypes
# Normalize can be set to True to print proportions instead of number 

df['Survived'].value_counts(normalize = True)
df['Pclass'].value_counts()
df['Sex'].value_counts()
df['Age'].value_counts()
df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Fare'].value_counts()
df['Cabin'].value_counts()
df['Embarked'].value_counts()
df['Ticket'].value_counts()
# Visualising Numerical features.

sns.distplot(df['Survived'])
# Visualising Numerical features and their corresponding boxplot.

plt.figure(1, figsize=(16,5))

plt.subplot(121)

sns.distplot(df['Pclass'])



plt.subplot(122)

sns.boxplot(y="Pclass", data = df)



plt.show()
plt.figure(1, figsize=(16,5))

plt.subplot(121)

sns.distplot(df['SibSp'])



plt.subplot(122)

sns.boxplot(y="SibSp", data = df)



plt.show()
# Print Percentiles for detecting outliers.

for i in range(0,100,10):

    var = df["SibSp"].values

    var = np.sort(var,axis = None)

    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))

print ("100 percentile value is ",var[-1])
for i in range(90,100):

    var = df["SibSp"].values

    var = np.sort(var,axis = None)

    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))

print ("100 percentile value is ",var[-1])
plt.figure(1, figsize=(16,5))

plt.subplot(121)

sns.distplot(df['Parch'])



plt.subplot(122)

sns.boxplot(y="Parch", data = df)



plt.show()
plt.figure(1, figsize=(16,5))

plt.subplot(121)

sns.distplot(df['Fare'])



plt.subplot(122)

sns.boxplot(y="Fare", data = df)



plt.show()
df1 = test.dropna()

plt.figure(1, figsize=(16,5))

plt.subplot(121)

sns.distplot(df1['Fare'])



plt.subplot(122)

sns.boxplot(y="Fare", data = df1)



plt.show()
df1 = df.dropna()

plt.figure(1, figsize=(16,5))

plt.subplot(121)

sns.distplot(df1['Age'])



plt.subplot(122)

sns.boxplot(y="Age", data = df1)



plt.show()
# Visualising Categorical Features.

plt.figure(1)

plt.subplot(121)

df['Sex'].value_counts(normalize=True).plot.bar(figsize=(16,5), title= 'Sex')



plt.subplot(122)

df['Embarked'].value_counts(normalize=True).plot.bar(figsize=(16,5), title= 'Embarked')
# Grouping Data by target variable and analysing features.

print(pd.crosstab(df['Pclass'],df['Survived']))

ct = pd.crosstab(df['Pclass'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
ct = pd.crosstab(df['SibSp'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
ct = pd.crosstab(df['Parch'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
ct = pd.crosstab(df['Sex'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
ct = pd.crosstab(df['Embarked'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
# Grouping Features using Quantile cut.

df['Age_bin'] = pd.qcut(df['Age'], 4)

df[['Age_bin', 'Survived']].groupby(['Age_bin'], as_index=False).mean().sort_values(by='Age_bin', ascending=True)
# Making bins for continous feature.

bins=[0,20,28,38,80]

group=[0,1,2,3]

df['Age_bin']=pd.cut(df['Age'],bins,labels=group)
ct = pd.crosstab(df['Age_bin'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
df['Fare_bin'] = pd.qcut(df['Fare'], 4)

df[['Fare_bin', 'Survived']].groupby(['Fare_bin'], as_index=False).mean().sort_values(by='Fare_bin', ascending=True)
bins=[0,8,15,32,513]

group=[0,1,2,3]

df['Fare_bin']=pd.cut(df['Fare'],bins,labels=group)
ct = pd.crosstab(df['Fare_bin'],df['Survived'])

ct.plot.bar(stacked=True)

plt.legend(title='Survived')
# Print correlation matrix

matrix = df.corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(matrix, vmax=.8, annot = True, square=True, cmap="BuPu");
# Checking the missing values

df.isnull().sum()
test.isnull().sum()
# Replacing null values with 0 and other values with 1.

# Feature that tells whether a passenger had a cabin on the Titanic

df['Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# Filling null values of categorical features with mode.

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Age_bin'].fillna(df['Age_bin'].mode()[0], inplace=True)

df['Fare_bin'].fillna(df['Fare_bin'].mode()[0], inplace=True)
# Grouping Embarked by target values.

df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Making pivot table of Age Feature by Pclass and Sex feature and taking median to fill null values.

table = df.pivot_table(values='Age', index='Pclass' ,columns='Sex', aggfunc=np.median)



# Define function to return value of this pivot_table

def fage(x):

 return table.loc[x['Pclass'],x['Sex']]



# Replace missing values

df['Age'].fillna(df[df['Age'].isnull()].apply(fage, axis=1), inplace=True)
# Making pivot table of Age Feature by Pclass and Sex feature and taking median to fill null values.

table = test.pivot_table(values='Age', index='Pclass' ,columns='Sex', aggfunc=np.median)



# Define function to return value of this pivot_table

def fage(x):

 return table.loc[x['Pclass'],x['Sex']]



# Replace missing values

test['Age'].fillna(test[test['Age'].isnull()].apply(fage, axis=1), inplace=True)
# Making pivot table of Age Feature by Pclass and Sex feature and taking median to fill null values.

table = test.pivot_table(values='Fare', index='Pclass' ,columns='Sex', aggfunc=np.median)



# Define function to return value of this pivot_table

def fage(x):

 return table.loc[x['Pclass'],x['Sex']]



# Replace missing values

test['Fare'].fillna(test[test['Fare'].isnull()].apply(fage, axis=1), inplace=True)
# Importing packages for model training.

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier
# Dropping unneccesary features.

df = df.drop(['PassengerId','Ticket','Name','Fare_bin','Age_bin'],axis=1)

test = test.drop(['PassengerId','Ticket','Name'],axis=1)
# Preparing Independent and dependent features.

X = df.drop('Survived',1)

y = df.Survived
# Adding dummies to the dataset

X = pd.get_dummies(X)

test=pd.get_dummies(test)
X.shape, test.shape
# Splitting Data into test train data for cross validation.

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size =0.2, random_state=1)
# logistic regression using K-fold cross validation

i=1

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X,y):

     print('\n{} of kfold {}'.format(i,kf.n_splits))

     xtr,xvl = X.loc[train_index],X.loc[test_index]

     ytr,yvl = y[train_index],y[test_index]

    

     model = LogisticRegression(random_state=1)

     model.fit(xtr, ytr)

     pred_test = model.predict(xvl)

     score = accuracy_score(yvl,pred_test)

     print('accuracy_score',score)

     i+=1

pred=model.predict_proba(xvl)[:,1]
# Roc Curve

from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(yvl,  pred)

auc = metrics.roc_auc_score(yvl, pred)

plt.figure(figsize=(12,8))

plt.plot(fpr,tpr,label="validation, auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc=4)

plt.show()
model = LogisticRegression()

model.fit(X, y)

acc_log = round(model.score(X, y) * 100, 2)

acc_log
model = LogisticRegression()

model.fit(X_train, y_train)

acc_log = round(model.score(X_train, y_train) * 100, 2)

acc_log
acc_logv = round(model.score(X_valid, y_valid) * 100, 2)

acc_logv
coeff_df = pd.DataFrame(df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(model.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()

svc.fit(X, y)

pred_test = svc.predict(test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc
acc_svcv = round(svc.score(X_valid, y_valid) * 100, 2)

acc_svcv
# Read submission file

submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submission['Survived'] = pred_test               # Fill predictions in Survived variable of submission file

submission['PassengerId'] = test_original['PassengerId']    # Fill Passenger Id of submission file with the Passenger Id of original test file
# Converting submission file to .csv format

pd.DataFrame(submission, columns=['PassengerId','Survived']).to_csv('SVM.csv')
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
acc_knnv = round(knn.score(X_valid, y_valid) * 100, 2)

acc_knnv
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
acc_gaussianv = round(gaussian.score(X_valid, y_valid) * 100, 2)

acc_gaussianv
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
acc_linear_svcv = round(linear_svc.score(X_valid, y_valid) * 100, 2)

acc_linear_svcv
sgd = SGDClassifier()

sgd.fit(X_train, y_train)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd
acc_sgdv = round(sgd.score(X_valid, y_valid) * 100, 2)

acc_sgdv
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
acc_decision_treev = round(decision_tree.score(X_valid, y_valid) * 100, 2)

acc_decision_treev
random_forest = RandomForestClassifier(n_estimators=100, max_depth = 2)

random_forest.fit(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
acc_random_forestv = round(random_forest.score(X_valid, y_valid) * 100, 2)

acc_random_forestv
df['Fare'] = np.log(df['Fare'] + 1)
X = df.drop('Survived',1)

y = df.Survived
X=pd.get_dummies(X)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size =0.2, random_state=1)
i=1

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X,y):

     print('\n{} of kfold {}'.format(i,kf.n_splits))

     xtr,xvl = X.loc[train_index],X.loc[test_index]

     ytr,yvl = y[train_index],y[test_index]

    

     model = LogisticRegression(random_state=1)

     model.fit(xtr, ytr)

     pred_test = model.predict(xvl)

     score = accuracy_score(yvl,pred_test)

     print('accuracy_score',score)

     i+=1
model = LogisticRegression()

model.fit(X, y)

acc_log = round(model.score(X, y) * 100, 2)

acc_log
model = LogisticRegression()

model.fit(X_train, y_train)

acc_log1 = round(model.score(X_train, y_train) * 100, 2)

acc_log1
acc_logv1 = round(model.score(X_valid, y_valid) * 100, 2)

acc_logv1
coeff_df = pd.DataFrame(df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(model.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()

svc.fit(X, y)

acc_svc1 = round(svc.score(X_train, y_train) * 100, 2)

acc_svc1
acc_svcv1 = round(svc.score(X_valid, y_valid) * 100, 2)

acc_svcv1
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

acc_knn1 = round(knn.score(X_train, y_train) * 100, 2)

acc_knn1
acc_knnv1 = round(knn.score(X_valid, y_valid) * 100, 2)

acc_knnv1
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

acc_gaussian1 = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian1
acc_gaussianv1 = round(gaussian.score(X_valid, y_valid) * 100, 2)

acc_gaussianv1
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

acc_linear_svc1 = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc1
acc_linear_svcv1 = round(linear_svc.score(X_valid, y_valid) * 100, 2)

acc_linear_svcv1
sgd = SGDClassifier()

sgd.fit(X_train, y_train)

acc_sgd1 = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd1
acc_sgdv1 = round(sgd.score(X_valid, y_valid) * 100, 2)

acc_sgdv1
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

acc_decision_tree1 = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree1
acc_decision_treev1 = round(decision_tree.score(X_valid, y_valid) * 100, 2)

acc_decision_treev1
random_forest = RandomForestClassifier(n_estimators=100, max_depth = 2)

random_forest.fit(X_train, y_train)

acc_random_forest1 = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest1
acc_random_forestv1 = round(random_forest.score(X_valid, y_valid) * 100, 2)

acc_random_forestv1
# Accuracy of Diifferent algorithms before feature engineering

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score_tr': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian,

              acc_sgd, acc_linear_svc, acc_decision_tree],

    'Score_cv': [acc_svcv, acc_knnv, acc_logv, 

              acc_random_forestv, acc_gaussianv,

              acc_sgdv, acc_linear_svcv, acc_decision_treev]})

models.sort_values(by='Score_cv', ascending=False)
# Accuracy of Diifferent algorithms after feature engineering

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score_tr': [acc_svc1, acc_knn1, acc_log1, 

              acc_random_forest1, acc_gaussian1,

              acc_sgd1, acc_linear_svc1, acc_decision_tree1],

    'Score_cv': [acc_svcv1, acc_knnv1, acc_logv1, 

              acc_random_forestv1, acc_gaussianv1,

              acc_sgdv1, acc_linear_svcv1, acc_decision_treev1]})

models.sort_values(by='Score_cv', ascending=False)