# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_df_y = pd.read_csv('../input/gender_submission.csv')
train_df.head()
train_df.describe()
train_df.info()
train_df.isnull().sum()
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace = True)
train_df.sample(10)

train_df['Title'] = train_df.Name.str.extract('([A-Za-z]+)\.')

#pd.crosstab(train_df['Title'], train_df['Title'].count())

train_df['Title'] = train_df['Title'].apply(lambda x: 'Misc' if ((train_df['Title']==x).sum() < 8) else x)
#pd.crosstab(train_df['Title'],train_df['Title'].count() )

#train_df['Title'].unique()
train_df['FamilySize'] = train_df ['SibSp'] + train_df['Parch'] + 1
train_df['AgeBin'] = pd.cut(train_df['Age'].astype(int), 5)
train_df['FareBin'] = pd.qcut(train_df['Fare'], 4)

train_df['IsAlone'] = 1
train_df['IsAlone'] = train_df['FamilySize'].apply(lambda x: 0 if x > 1 else 1)
drop_column = ['Name','SibSp', 'Parch','Age','Fare','PassengerId','Cabin', 'Ticket']
train_df.drop(drop_column, axis=1, inplace = True)
train_df.sample(10)
#train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train_df[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['FareBin', 'Survived']].groupby(['FareBin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
visualization_df = train_df.copy()
# Encoding categorical data
label = LabelEncoder()
#onehotencoder = OneHotEncoder()
visualization_df['Sex_Code'] = label.fit_transform(visualization_df['Sex'])
#converted_df = onehotencoder.fit_transform(converted_df['Sex_Code']).toarray()
visualization_df['Embarked_Code'] = label.fit_transform(visualization_df['Embarked'])
visualization_df['Title_Code'] = label.fit_transform(visualization_df['Title'])
visualization_df['AgeBin_Code'] = label.fit_transform(visualization_df['AgeBin'])
visualization_df['FareBin_Code'] = label.fit_transform(visualization_df['FareBin'])

sns.set(rc={'figure.figsize':(14,12)})
p =sns.heatmap(visualization_df.corr(), annot=True, cmap=sns.diverging_palette(220, 10, as_cmap=True))

# Feature Scaling in case we have any continues variable
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
column_name = ['Sex','Pclass', 'Embarked', 'Title', 'FamilySize', 'AgeBin', 'FareBin']

converted_df = pd.get_dummies(train_df, columns=column_name,drop_first=True)

x_data = converted_df.iloc[:, 1:].values
y_data = converted_df.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20, random_state = 0)

converted_df.head()
# Logistic Regression

logreg = linear_model.LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)


#acc_log = round(logreg.score(y_test, y_pred) * 100, 2)


#accuracy score
acc_log = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
acc_log


# Making the Confusion Matrix
#cm = metrics.confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot=True,annot_kws={"size": 16})# font size
# Support Vector Machines

svc = svm.SVC(kernel = 'rbf', gamma=0.085 ,random_state = 0)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

acc_svc = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)

acc_svc
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

acc_knn = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = naive_bayes.GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)

acc_gaussian = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
acc_gaussian
# Decision Tree

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)

acc_decision_tree
# Random Forest

random_forest = ensemble.RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_random_forest = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)

acc_random_forest
#xgboost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

# Predicting the Test set results
y_pred = xgb.predict(x_test)

acc_xgb = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)

acc_xgb
# Applying k-Fold Cross Validation
#accuracies = model_selection.cross_val_score(estimator = svc, X = x_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()

# Applying Grid Search to find the best model and the best parameters
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.075, 0.080, 0.085, 0.090, 0.095]}]
grid_search = model_selection.GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
grid_svm_acc = grid_search.best_score_

grid_svm_acc = round(grid_svm_acc * 100, 2)

best_parameters = grid_search.best_params_
best_parameters
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Decision Tree','xgboost','Grid Search'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_decision_tree,acc_xgb,grid_svm_acc]})
models.sort_values(by='Score', ascending=False)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.')

x_test = test_df[['PassengerId','Name']]
x_test['FamilySize'] = test_df ['SibSp'] + test_df['Parch'] + 1
x_test['IsAlone']= 1
x_test['IsAlone'] = x_test['FamilySize'].apply(lambda x: 0 if x > 1 else 1)

x_test['Sex_male'] = test_df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
x_test['Pclass_2'] = test_df['Pclass'].apply(lambda x: 1 if x == 2 else 0)
x_test['Pclass_3'] = test_df['Pclass'].apply(lambda x: 1 if x == 3 else 0)
x_test['Embarked_Q'] = test_df['Embarked'].apply(lambda x: 1 if x == 'Q' else 0)
x_test['Embarked_S'] = test_df['Embarked'].apply(lambda x: 1 if x == 'S' else 0)
x_test['Title_Misc'] = test_df['Title'].apply(lambda x: 0 if x in ('Mr', 'Mrs', 'Miss', 'Master') else 1)
x_test['Title_Miss'] = test_df['Title'].apply(lambda x: 1 if x == 'Miss' else 0)
x_test['Title_Mr'] = test_df['Title'].apply(lambda x: 1 if x == 'Mr' else 0)
x_test['Title_Mrs'] = test_df['Title'].apply(lambda x: 1 if x == 'Mrs' else 0)
x_test['FamilySize_2'] = x_test['FamilySize'].apply(lambda x: 1 if x == 2 else 0)
x_test['FamilySize_3'] = x_test['FamilySize'].apply(lambda x: 1 if x == 3 else 0)
x_test['FamilySize_4'] = x_test['FamilySize'].apply(lambda x: 1 if x == 4 else 0)
x_test['FamilySize_5'] = x_test['FamilySize'].apply(lambda x: 1 if x == 5 else 0)
x_test['FamilySize_6'] = x_test['FamilySize'].apply(lambda x: 1 if x == 6 else 0)
x_test['FamilySize_7'] = x_test['FamilySize'].apply(lambda x: 1 if x == 7 else 0)
x_test['FamilySize_8'] = x_test['FamilySize'].apply(lambda x: 1 if x == 8 else 0)
x_test['FamilySize_11'] = x_test['FamilySize'].apply(lambda x: 1 if x > 8 else 0)
x_test['AgeBin_(16.0, 32.0]'] = test_df['Age'].apply(lambda x: 1 if (x > 16 and x <= 32) else 0)
x_test['AgeBin_(32.0, 48.0]'] = test_df['Age'].apply(lambda x: 1 if (x > 32 and x <= 48) else 0)
x_test['AgeBin_(48.0, 64.0]'] = test_df['Age'].apply(lambda x: 1 if (x > 48 and x <= 64) else 0)
x_test['AgeBin_(64.0, 80.0]'] = test_df['Age'].apply(lambda x: 1 if (x > 64 and x <= 80) else 0)
x_test['FareBin_(7.91, 14.454]'] = test_df['Fare'].apply(lambda x: 1 if (x > 7.91 and x <= 14.454) else 0)
x_test['FareBin_(14.454, 31.0]'] = test_df['Fare'].apply(lambda x: 1 if (x > 14.454 and x <= 31.0) else 0)
x_test['FareBin_(31.0, 512.329]'] = test_df['Fare'].apply(lambda x: 1 if (x > 31.0 and x <= 52.329) else 0)



drop_column = ['PassengerId','Name','FamilySize']
x_test.drop(drop_column, axis=1, inplace = True)
x_test.head()
x_test = x_test.iloc[:, :].values
y_test = test_df_y.iloc[:, 1].values
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
acc_decision_tree