import numpy as np 

import pandas as pd 

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

import statsmodels.api as sm



import warnings

warnings.filterwarnings('ignore')



import plotly.express as px

import plotly.graph_objects as go

import plotly.subplots as make_subplots
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head(2)
train_df.isnull().sum()
test_df.isnull().sum()
df = pd.concat([train_df, test_df], axis = 0, sort = False)
print(f'Combined shape = {df.shape}')

print(f'Test shape = {test_df.shape}')

print(f'Train shape = {train_df.shape}')
df = df.reset_index()
df.drop(columns = ['index'], inplace = True)
df.info()
df.describe()
null_cols = df.isnull().sum()

null_cols = pd.DataFrame(null_cols)

null_cols = null_cols.reset_index()

null_cols = null_cols.rename(columns = {'index' : 'Features', 0 : '# of Missing Values'})

null_cols
fig = px.bar(null_cols, 

            x = 'Features',

            y = '# of Missing Values',

            hover_data = ['# of Missing Values'],

            color = 'Features',

            title = 'Missing values in columns')

fig.show()
Sex = pd.DataFrame(df['Sex'].value_counts())

Sex = Sex.reset_index().rename(columns = {'index' : 'Gender', 'Sex' : 'Count'})

fig = px.pie(Sex,

            values = 'Count',

            names = 'Gender',

            title = '# of Male and Female in dataset')

fig.show()
Embarked = pd.DataFrame(df['Embarked'].value_counts())

Embarked = Embarked.reset_index().rename(columns = {'index' : 'Embarked', 'Embarked' : 'Count'})

fig = px.pie(Embarked,

            values = 'Count',

            names = 'Embarked',

            title = 'Embarkment')

fig.show()
PassengerClass = pd.DataFrame(df['Pclass'].value_counts())

PassengerClass = PassengerClass.reset_index().rename(columns = {'index' : 'Pclass', 'Pclass' : 'Count'})

fig = px.pie(PassengerClass,

            values = 'Count',

            names = 'Pclass',

            title = '# of passengers in different class')

fig.show()
df['Age'].fillna(df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median())), inplace = True)

df.isnull().sum()
P = pd.DataFrame()

P['age_range'] = pd.cut(df['Age'], [0,20,40,60,100], include_lowest=True)

AgeRange = pd.DataFrame(P['age_range'].value_counts())

AgeRange = AgeRange.reset_index().rename(columns = {'index' : 'Range', 'age_range' : 'Count'})

plt.pie(AgeRange['Count'],

       labels=AgeRange['Range'],

       autopct='%.3f%%')

plt.title('Age bracket Comparison')

plt.show()
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

df.isna().sum()
df['Fare'].fillna(df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median())), inplace = True)

df.isnull().sum()
df.drop(columns = ['Cabin'], inplace = True)
df.isnull().sum()
Classwise_Embarkment = pd.DataFrame(df[['PassengerId','Pclass', 'Embarked']].groupby(['Pclass','Embarked']).count())

Classwise_Embarkment
ClasswiseGender=pd.DataFrame(df[['Pclass', 'PassengerId', 'Sex']].groupby(['Pclass', 'Sex']).count()).rename(columns = {'PassengerId' : 'Count'})

ClasswiseGender
df.drop(columns = ['Ticket'], inplace = True)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

df['Title'].value_counts()
title_list = list(df['Title'].unique())

list_rare = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

for title in df['Title']:

    df['Title'] = df['Title'].replace(list_rare, 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['Title'] = df['Title'].replace('Miss', 'Miss')

    df['Title'] = df['Title'].replace('Mrs', 'Mrs')

    df['Title'] = df['Title'].replace('Mr', 'Mr')

    df['Title'] = df['Title'].replace('Master', 'Master')

df['Title'].value_counts()
df['Accompany'] = df['SibSp']+df['Parch']
df[['Pclass', 'Accompany', 'PassengerId']].groupby(['Pclass', 'Accompany']).count()
df.drop(columns = ['Name', 'SibSp', 'Parch'], inplace = True)

df.reset_index(inplace = True)
gender_survival = df[['Sex', 'Survived']].groupby('Survived').count()

gender_survival = gender_survival.reset_index().rename(columns = {'index' : 'Survived'})

gender_survival
labels = ['Female', 'Male']

fig = px.pie(gender_survival,

            values = 'Sex',

            names = labels,

            title = 'Genderwise chance of Survival')

fig.show()
Classwise_survival = df[['Pclass', 'Survived', 'PassengerId']].groupby(['Pclass', 'Survived']).count()

Classwise_survival = Classwise_survival.reset_index().rename(columns = {'index' : 'Pclass'})

Classwise_survival
Class1 = Classwise_survival[Classwise_survival['Pclass'] == 1]

Class2 = Classwise_survival[Classwise_survival['Pclass'] == 2]

Class3 = Classwise_survival[Classwise_survival['Pclass'] == 3]

fig = go.Figure(data=[

    go.Bar(name='Pclass1', x=Class1['Survived'], y=Class1['PassengerId']),

    go.Bar(name='Pclass2', x=Class2['Survived'], y=Class2['PassengerId']),

    go.Bar(name='Pclass3', x=Class3['Survived'], y=Class3['PassengerId'])

])

# Change the bar mode

fig.update_layout(barmode='group', title = 'Classwise Chance of survival')

fig.show()
Embarked = pd.DataFrame(df['Embarked'].value_counts())

Embarked.reset_index().rename(columns = {'index' : 'E'})
# from sklearn.preprocessing import LabelEncoder

# df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# df['Sex'].value_counts()

# df.head()

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder() 

data = pd.DataFrame(onehotencoder.fit_transform(df[['Sex']]).toarray())

data = data.reset_index()

df = pd.merge(left = df, right = data, on = 'index')

#test_df.rename(columns = {0:'Master', 1:'Miss', 2:'Mr', 3:'Mrs', 4:'Rare'}, inplace = True)

df = df.drop(columns = ['Sex'])

df.sort_values(by = 'PassengerId', inplace = True)
df.rename(columns = {0:'Female', 1:'Male'}, inplace= True)
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder() 

data = pd.DataFrame(onehotencoder.fit_transform(df[['Embarked']]).toarray())

data = data.reset_index()

df = pd.merge(left = df, right = data, on = 'index')

df.rename(columns = {0 : 'C', 1 : 'Q', 2 : 'S'}, inplace = True)

df.drop(columns = 'Embarked', inplace = True)
onehotencoder = OneHotEncoder() 

data = pd.DataFrame(onehotencoder.fit_transform(df[['Title']]).toarray())

data = data.reset_index()

df = pd.merge(left = df, right = data)

df.rename(columns = {0:'Master', 1:'Miss', 2:'Mr', 3:'Mrs', 4:'Rare'}, inplace = True)

df = df.sort_values(by = 'PassengerId', ascending = True)
df.drop(columns = 'Title', inplace = True)
df['Fare'] = np.log1p(df['Fare'])
# from sklearn.feature_selection import SelectKBest, f_classif



# predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "Accompany"]



# # Perform feature selection

# selector = SelectKBest(f_classif, k=5)

# selector.fit(train_df[predictors], train_df["Survived"])



# # Get the raw p-values for each feature, and transform from p-values into scores

# scores = -np.log10(selector.pvalues_)



# # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?

# plt.bar(range(len(predictors)), scores)

# plt.xticks(range(len(predictors)), predictors, rotation='vertical')

# plt.show()
df.drop(columns = ['index'], inplace = True)
df['Age'] = pd.cut(df['Age'], 5)
from sklearn.preprocessing import LabelEncoder

df['Age'] = LabelEncoder().fit_transform(df['Age'])

df['Age'].value_counts()
test_df = df[df['PassengerId'] > 891]

test_df.shape
train_df = df[df['PassengerId'] <= 891]

train_df.shape
test_df.isnull().sum()
test_df.drop(columns = ['Survived'], inplace = True)
passenger_id = test_df['PassengerId']

test_df.drop(columns = ['PassengerId'], inplace = True)
test_df.isnull().sum()
x_train = train_df.drop(columns = 'Survived')

y_train = train_df['Survived']

x_train.drop(columns = ['PassengerId'], inplace = True)
y_train = pd.DataFrame(y_train)
y_train = y_train['Survived'].astype(int)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(test_df)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression()

grid_values = {'penalty': ['l1', 'l2'],

               'C':[1,3,5,7,10,11,12]

              }

grid = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall', cv = 5)

grid.fit(x_train, y_train)
grid.best_params_
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C= 3, penalty= 'l2')

logreg.fit(x_train, y_train)

y_pred = logreg.predict(test_df)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
# from sklearn.model_selection import cross_val_score

# clf = LogisticRegression(C= 10, penalty= 'l1', solver= 'liblinear')

# scores = cross_val_score(clf, x_train, y_train, cv=5)

# scores = scores.mean()

# scores
from sklearn.model_selection import cross_val_score

clf = LogisticRegression(C= 3, penalty= 'l2')

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
# Support Vector Machines

svc = SVC(C= 3, gamma= 0.06, kernel= 'linear')

svc.fit(x_train, y_train)

Y_pred = svc.predict(test_df)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
# from sklearn.model_selection import GridSearchCV 

# param_grid = {'C': [1, 3, 5, 7, 9], 

#               'gamma': [0.06, 0.08, 0.1, 0.12, 0.14], 

#               'kernel': ['rbf']}  

  

# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

  

# # fitting the model for grid search 

# grid.fit(x_train, y_train) 
# grid.best_params_
# from sklearn.model_selection import cross_val_score

# clf = SVC(C= 2, gamma = 1, kernel= 'linear')

# scores = cross_val_score(clf, x_train, y_train, cv=5)

# scores = scores.mean()

# scores
from sklearn.model_selection import cross_val_score

clf = SVC(C= 3, gamma= 0.06, kernel= 'linear')

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
# knn = KNeighborsClassifier(n_neighbors = 3)

# knn.fit(x_train, y_train)

# Y_pred = knn.predict(test_df)

# acc_knn = round(knn.score(x_train, y_train) * 100, 2)

# acc_knn
# from sklearn.model_selection import GridSearchCV

# from sklearn.neighbors import KNeighborsClassifier

# #making the instance

# model = KNeighborsClassifier(n_jobs=-1)

# #Hyper Parameters Set

# params = {'n_neighbors':[3,4,5,6,7,8,9,10],

#           'leaf_size':[10,20,30,40,50,60],

#           'weights':['uniform', 'distance'],

#           'algorithm':['auto', 'ball_tree','kd_tree','brute'],

#           'n_jobs':[-1]}

# #Making models with hyper parameters sets

# model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

# #Learning

# model1.fit(x_train,y_train)
# model1.best_params_
knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 40, n_neighbors = 7, weights = 'uniform')

knn.fit(x_train, y_train)

Y_pred = knn.predict(test_df)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

acc_knn
# from sklearn.model_selection import cross_val_score

# clf = KNeighborsClassifier()

# scores = cross_val_score(clf, x_train, y_train, cv=5)

# scores = scores.mean()

# scores
from sklearn.model_selection import cross_val_score

clf = KNeighborsClassifier(algorithm = 'auto', leaf_size = 40, n_neighbors = 7, weights = 'uniform')

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

Y_pred = gaussian.predict(test_df)

acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

acc_gaussian
from sklearn.model_selection import cross_val_score

clf = GaussianNB()

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
# decision_tree = DecisionTreeClassifier()

# decision_tree.fit(x_train, y_train)

# Y_pred = decision_tree.predict(test_df)

# acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

# acc_decision_tree
# from sklearn.model_selection import GridSearchCV

# from sklearn.tree import DecisionTreeClassifier

# #making the instance

# model= DecisionTreeClassifier(random_state=0)

# #Hyper Parameters Set

# params = {

#           'max_features': ['auto', 'sqrt', 'log2'],

#           'min_samples_split': [10,15,20,25,30,35,40,45,50,55,60,65,70], 

#           'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],

#           }



# model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

# #Learning

# model1.fit(x_train,y_train)
# model1.best_params_
decision_tree = DecisionTreeClassifier(max_features= 'auto', min_samples_leaf= 2, min_samples_split= 20)

decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(test_df)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(max_features= 'auto', min_samples_leaf= 2, min_samples_split= 20)

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(max_features= 'auto', min_samples_leaf= 2, min_samples_split= 20)

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
# random_forest = RandomForestClassifier(n_estimators=100)

# random_forest.fit(x_train, y_train)

# Y_pred = random_forest.predict(test_df)

# random_forest.score(x_train, y_train)

# acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

# acc_random_forest
# from sklearn.model_selection import GridSearchCV

# from sklearn.ensemble import RandomForestClassifier

# #making the instance

# model=RandomForestClassifier()

# #hyper parameters set

# params = {'criterion':['gini','entropy'],

#           'n_estimators':[11,13,15,17,19,21],

#           'min_samples_leaf':[1,2,3,4,5,6],

#           'min_samples_split':[2,3,4,5,6],

#           'n_jobs':[-1]}

# #Making models with hyper parameters sets

# model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

# #learning

# model1.fit(x_train,y_train)
# model1.best_params_
random_forest = RandomForestClassifier(criterion= 'gini',

                                       min_samples_leaf= 6,

                                       min_samples_split= 6,

                                       n_estimators= 21,

                                       n_jobs= -1)

random_forest.fit(x_train, y_train)

Y_pred = random_forest.predict(test_df)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

acc_random_forest
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(criterion= 'gini',

                            min_samples_leaf= 6,

                            min_samples_split= 6,

                            n_estimators= 21,

                            n_jobs= -1)

scores = cross_val_score(clf, x_train, y_train, cv=5)

scores = scores.mean()

scores
# from sklearn.model_selection import cross_val_score

# clf = RandomForestClassifier(n_estimators=100)

# scores = cross_val_score(clf, x_train, y_train, cv=5)

# scores = scores.mean()

# scores
from sklearn.ensemble import VotingClassifier

ensemble=VotingClassifier(estimators=[('Decision Tree', decision_tree), 

                                      ('Random Forest', random_forest), 

                                      ('KNN', knn), 

                                      ('SVC', svc), 

                                      ('Logistic Regression', logreg)], 

                                      voting='soft', 

                                      weights=[1,1,2,1,1]).fit(x_train,y_train)

scores = cross_val_score(clf, x_train, y_train, cv=10)

acc_voting = scores.mean()*100

acc_voting





# from mlxtend.classifier import StackingClassifier

# clf = StackingClassifier(classifiers=[random_forest, decision_tree, knn, svc, logreg], 

#                           meta_classifier=logreg)

# scores = cross_val_score(clf, x_train, y_train, cv=10)

# acc_stacking = scores.mean()*100

# acc_stacking

# #, gaussian







# # for clf, label in zip([random_forest, decision_tree, gaussian, knn, svc, logreg], ['Random Forest', 'Decision Tree', 'GaussianNB', 'KNN', 'SVC', 'Logistic Regression']):

# #     scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')

# #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
passenger_id = pd.DataFrame(passenger_id)

passenger_id.head(2)
test_df.reset_index(inplace = True)

test_df.head(2)
test_df['PassengerId'] = test_df['index']+1

test_df.head(2)
test_df.tail(2)
train_df.reset_index()

test_df.reset_index()

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)