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
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns



from matplotlib import pyplot as plt
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

train_data_copy1 = train_data.copy(deep = True)

data_combined = [train_data_copy1, test_data]

train_data.info()
train_data.sample(10)
print('Train columns with null values:\n', train_data.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', test_data.isnull().sum())

print("-"*10)



train_data.describe(include = 'all')
for dataset in data_combined:

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)    
drop_column = ['PassengerId', 'Cabin', 'Ticket']

train_data_copy1.drop(drop_column, axis =1, inplace = True)

print(train_data_copy1.isnull().sum())

print('*'*10)

print(test_data.isnull().sum())
fig = px.histogram(train_data_copy1, x = 'Age', title = "Age Distribution to Gender", color = 'Sex',color_discrete_sequence = px.colors.qualitative.Safe, nbins = 15)

fig.show()

sns.boxplot(data = train_data_copy1, x = 'Age')
fig = px.histogram(train_data_copy1, x = 'Age', title = "Age Distribution to Survived Passenger", color = 'Survived',color_discrete_sequence = px.colors.qualitative.Set2, nbins = 15)

fig.show()
fig = px.pie(train_data_copy1, values = 'Survived', names= 'Sex', title = "Pie Chart: Gender wise survival")

fig.show()
fig = px.pie(train_data_copy1, values = 'Survived', names= 'Pclass', title = "Pie Chart: Class wise survival",color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
fig = px.pie(train_data_copy1, values = 'Survived', names= 'Embarked', title = "Pie Chart: Survival according to Embarked Location",color_discrete_sequence=px.colors.qualitative.Bold)

fig.show()
plt.figure(figsize = (10,10))

plt.title('Swarm Plot: Survived to Age and Gender', fontsize = 15)

sns.swarmplot(x = 'Survived', y = 'Age',hue = 'Sex',data = train_data_copy1, size = 5)

sns.pairplot(train_data_copy1, vars = ['Age','Fare','Pclass'], hue = 'Survived', palette = 'Set1')



for dataset in data_combined:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #(Including self as 1)

    dataset['IsAlone']= 1

    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0 # Assign 0 if familysize greater than 1

    

    dataset['Title'] = dataset['Name'].str.split(',',expand = True)[1].str.split(".",expand = True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'],4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int),5)

    

title_count = 10

title_names = (train_data_copy1['Title'].value_counts()<title_count)





train_data_copy1['Title'] = train_data_copy1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)        

print(train_data_copy1['Title'].value_counts())



print('*'*10)

train_data_copy1.info()

test_data.info()

train_data_copy1.sample(10)



    

       
plt.figure(figsize = (10,8))

plt.title('Count Plot: Survived to IsAlone', fontsize = 15)

sns.countplot(data = train_data_copy1,x = 'Survived', hue = 'IsAlone', palette = 'Set1')





plt.figure(figsize = (10,8))

plt.title('Count Plot: Survived to IsAlone', fontsize = 15)

sns.countplot(data = train_data_copy1,x = 'FamilySize', hue = 'Survived', palette = 'muted')





la = LabelEncoder()

for dataset in data_combined:

    

    dataset['Sex_Code'] = la.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = la.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = la.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = la.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = la.fit_transform(dataset['FareBin'])

Target = ['Survived']





#define x variables for original features aka feature selection

train_data_copy1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

train_data_copy1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

train_data_copy1_xy =  Target + train_data_copy1_x

print('Original X Y: ', train_data_copy1_xy, '\n')





#define x variables for original w/bin features to remove continuous variables

train_data_copy1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

train_data_copy1_xy_bin = Target + train_data_copy1_x_bin

print('Bin X Y: ', train_data_copy1_xy_bin, '\n')





#define x and y variables for dummy features original

train_data_copy1_dummy = pd.get_dummies(train_data_copy1[train_data_copy1_x])

train_data_copy1_x_dummy = train_data_copy1_dummy.columns.tolist()

train_data_copy1_xy_dummy = Target + train_data_copy1_x_dummy

print('Dummy X Y: ', train_data_copy1_xy_dummy, '\n')







train_data_copy1_dummy.head()



print('Train columns with null values: \n', train_data_copy1.isnull().sum())

print("-"*10)

print (train_data_copy1.info())

print("-"*10)



print('Test/Validation columns with null values: \n', test_data.isnull().sum())

print("-"*10)

print (test_data.info())

print("-"*10)



train_data.describe(include = 'all')
Y_train = train_data_copy1["Survived"]

X_train = pd.get_dummies(train_data_copy1[train_data_copy1_x_bin])

X_test = pd.get_dummies(test_data[train_data_copy1_x_bin])



Y_train.shape, X_train.shape, X_test.shape

                         
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print("Logistic Regression Score:",acc_log)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print("Naive-Bayes accuracy : ",acc_gaussian)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print ("KNeighbors accuracy score : ",acc_knn)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(data_train, target_train.values.ravel())

pred = random_forest.predict(data_test)

acc_random_forest =accuracy_score(target_test, pred)

print ("Random Forest accuracy score : ",acc_random_forest)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print ("Stochastic accuracy score : ",acc_sgd)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print ("Decision Tree accuracy score : ", acc_decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print ("Random Forest accuracy score : ",acc_random_forest)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, 

              acc_sgd, acc_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)





output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")