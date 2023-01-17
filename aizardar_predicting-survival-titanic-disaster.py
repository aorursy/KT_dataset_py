# Let's import some libraries



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# To load the image from a website



from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")
# Let's load our training and test data



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv").set_index('PassengerId')

train = pd.read_csv("../input/titanic/train.csv").set_index('PassengerId')

data = pd.concat([train, test], axis=0, sort=False)
train
from pandas_profiling import ProfileReport



profile = ProfileReport(train, title="Titanic Dataset")
profile.to_notebook_iframe()
train.describe()
# Let's print few lines of our training dataset



train.head()
# Let's define a function to create a bar chart. 



def bar_chart(feature):

    survived = data[data['Survived']==1][feature].value_counts()

    dead = data[data['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived, dead])

    df.index = ['Survived','Dead']

    df.plot(kind = 'bar', stacked = True, figsize = (15,10), fontsize=18)
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
data.isnull().sum()
print(data.Embarked.value_counts())

data.Embarked = data.Embarked.fillna('S')

data.Embarked.isnull().sum()
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

data.Title.value_counts()
data['Title'] = np.where((data.Title=='Capt') | (data.Title == 'Jonkheer') | (data.Title == 'Countess') | (data.Title == 'Dona') | (data.Title == 'Don') | (data.Title == 'Lady') | (data.Title == 'Major') | (data.Title == 'Rev') | (data.Title == 'Dr') | (data.Title == 'Sir') | (data.Title == 'Col'), 'other', data.Title)



# Lets also take care of Mlle, Ms. and Mme



data['Title'] = data['Title'].replace('Ms','Miss')

data['Title'] = data['Title'].replace('Mlle','Miss')

data['Title'] = data['Title'].replace('Mme','Mrs')

data['Title']

data.Title.value_counts()
train.head()
bar_chart('Title')
data["Age"].fillna(data.groupby("Title")["Age"].transform("median"), inplace=True)
data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

facet = sns.FacetGrid(data = data, hue = "Title", legend_out=True, size = 4.5)

facet = facet.map(sns.kdeplot, "Age")

facet.add_legend();
facet = sns.FacetGrid(data, hue="Survived",aspect=3)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, data['Age'].max()))

facet.add_legend()



plt.show()
data["Fare"].fillna(data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

data.head(25)
data.Cabin.value_counts()
data.groupby('Pclass').Cabin.value_counts()
data['Cabin'] = data['Cabin'].str[:1]

data.groupby('Pclass').Cabin.value_counts()
data.Cabin = data.Cabin.fillna('U')

data.groupby('Pclass').Cabin.value_counts()
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
data['Cabin'] = np.where((data.Pclass == 1) & (data.Cabin == 'U'), 'C',data.Cabin)

data['Cabin'] = np.where((data.Pclass == 2) & (data.Cabin == 'U'), 'D',data.Cabin)

data['Cabin'] = np.where((data.Pclass == 3) & (data.Cabin == 'U'), 'G',data.Cabin)

data['Cabin'] = np.where((data.Cabin == 'T'), 'C',data.Cabin)

data.groupby('Pclass').Cabin.value_counts()

data['FamilySize'] = data.SibSp + data.Parch + 1 

data.FamilySize
facet = sns.FacetGrid(data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, data['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
# Let's visualize our data

data.head()

# Let's first drop few features



data.drop('Name', axis=1, inplace=True)

data.drop('SibSp', axis=1, inplace=True)

data.drop('Parch', axis=1, inplace=True)

data.drop('Ticket', axis=1, inplace=True)
data.head()
sex_mapping = {"male": 0, "female":1}

data['Sex'] = data['Sex'].map(sex_mapping)

data.head()
data.Cabin.value_counts()
cabin_mapping = {"A": 0.0, "B":0.5, "C": 1.0, "D":1.5, "E": 2.0, "F":2.5, "G": 3.0}

data['Cabin'] = data["Cabin"].map(cabin_mapping)

data.head()
data.Embarked.value_counts()
embarked_mapping = {"S": 0, "C":1, "Q": 2}

data['Embarked'] = data['Embarked'].map(embarked_mapping)

data.head()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "other": 3}

data['Title'] = data['Title'].map(title_mapping)

data.head()
data.loc[data['Age'] <= 10, 'Age'] = 0,

data.loc[(data['Age'] > 10) & (data['Age'] <= 18), 'Age'] = 1,

data.loc[(data['Age'] > 18) & (data['Age'] <= 36), 'Age'] = 2,

data.loc[(data['Age'] > 36) & (data['Age'] <= 65), 'Age'] = 3,

data.loc[ data['Age'] > 65, 'Age'] = 4



data.head()
facet = sns.FacetGrid(data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, data['Fare'].max()))

facet.add_legend()

 

plt.show()
facet = sns.FacetGrid(data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, data['Fare'].max()))

facet.add_legend()

plt.xlim(0,30)

plt.show()
facet = sns.FacetGrid(data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, data['Fare'].max()))

facet.add_legend()

plt.xlim(30,100)

plt.show()
data.loc[ data['Fare'] <= 17, 'Fare'] = 0,

data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30), 'Fare'] = 1,

data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100), 'Fare'] = 2,

data.loc[ data['Fare'] > 100, 'Fare'] = 3

data.head()
# Let's first get our original test set from the data



test_sets = data.iloc[train.index.max():test.index.max()]

test_sets = test_sets.reset_index(drop=True)

test_sets.info()
# Let's now get our training set



train_sets = data.iloc[0:train.index.max()]

train_sets = train_sets.reset_index(drop=True)

train_sets.info()
# features and target for training

feature_cols = test_sets.columns.values[test_sets.columns.values != 'Survived']



X_train = train_sets[feature_cols]

y_train = train_sets['Survived']



X_test = test_sets[feature_cols]

y_test = test_sets['Survived']
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz
# Tuning the DecisionTreeClassifier by the GridSearchCV

parameters = {'max_depth' : np.arange(2, 12, dtype=int),

              'min_samples_leaf' :  np.arange(1, 6, dtype=int)}

classifier = DecisionTreeClassifier(random_state=2020)

model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

model.fit(X_train, y_train)

best_parameters = model.best_params_

print(best_parameters)
model=DecisionTreeClassifier(max_depth = best_parameters['max_depth'], 

                             random_state = 1118)

model.fit(X_train, y_train)
from sklearn import tree
# plot tree

dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns, class_names=['0', '1'], 

                           filled=True, rounded=False,special_characters=True, precision=7) 

graph = graphviz.Source(dot_data)

graph 
y_pred = model.predict(X_test).astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# Saving the result

pd.DataFrame({'Survived': y_pred}, index=test.index).reset_index().to_csv('Survived_pred_DTC.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
# RFC = RandomForestClassifier(random_state = 2020)



# from pprint import pprint



# print ("Parameters currently in use : \n")

# pprint(RFC.get_params())
# Tuning the RFC by the GridSearchCV

# parameters = {'max_depth' : np.arange(2, 12, 2, dtype=int),

#               'n_estimators' :  np.arange(100, 4000, 500, dtype=int)}

# classifier = RandomForestClassifier(random_state=2020)

# model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

# model.fit(X_train, y_train)

# best_parameters = model.best_params_

# print(best_parameters)
# model=RandomForestClassifier(max_depth = best_parameters['max_depth'],n_estimators = best_parameters['n_estimators'], 

#                              random_state = 1118)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test).astype(int)

# print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# # Saving the result

# pd.DataFrame({'Survived': y_pred}, index=test.index).reset_index().to_csv('Survived_pred_RFC.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier
KFC = KNeighborsClassifier()



from pprint import pprint



print ("Parameters currently in use : \n")

pprint(KFC.get_params())
parameters = {'n_neighbors': np.arange(1,50,1)}



classifier = KNeighborsClassifier()

model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

model.fit(X_train, y_train)

best_parameters = model.best_params_

print(best_parameters)

model=KNeighborsClassifier(n_neighbors = best_parameters['n_neighbors'])

model.fit(X_train, y_train)

y_pred = model.predict(X_test).astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# Saving the result

pd.DataFrame({'Survived': y_pred}, index=test.index).reset_index().to_csv('Survived_pred_KFC.csv', index=False)
from sklearn.svm import SVC



model = SVC()



from pprint import pprint



print ("Parameters currently in use : \n")



pprint(model.get_params())
C = [0.001,0.01,0.1,1,10]

gamma = [1, 0.1, 0.01, 0.001, 0.0001]



parameters = {'C': C, 'gamma': gamma, 'kernel' : ['rbf']}



classifier = SVC()

model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

model.fit(X_train, y_train)

best_parameters = model.best_params_

print(best_parameters)
model = SVC(C = best_parameters['C'], gamma = best_parameters['gamma'], kernel = 'rbf')

model.fit(X_train, y_train)

y_pred = model.predict(X_test).astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# Saving the result

pd.DataFrame({'Survived': y_pred}, index=test.index).reset_index().to_csv('Survived_pred_SVC.csv', index=False)
from sklearn.ensemble import AdaBoostClassifier



model = AdaBoostClassifier()



from pprint import pprint



print ("Parameters currently in use : \n")



pprint(model.get_params())
# parameters = {'n_estimators': np.arange(10,2000,50), 'learning_rate' : [0.1, 0.5, 1.5]}



# classifier = AdaBoostClassifier()

# model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

# model.fit(X_train, y_train)

# best_parameters = model.best_params_

# print(best_parameters)
# model = AdaBoostClassifier(n_estimators = best_parameters['n_estimators'], learning_rate = best_parameters['learning_rate'],random_state = 2020)

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test).astype(int)

# print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# # Saving the result

# pd.DataFrame({'Survived': y_pred}, index=test.index).reset_index().to_csv('Survived_pred_AdaBoost.csv', index=False)
from xgboost.sklearn import XGBClassifier
model = XGBClassifier()



from pprint import pprint



print ("Parameters currently in use : \n")



pprint(model.get_params())
from sklearn.model_selection import RandomizedSearchCV



# Create the parameter grid: gbm_param_grid 

gbm_param_grid = {

    'n_estimators': range(8, 20),

    'max_depth': range(6, 10),

    'learning_rate': [.4, .45, .5, .55, .6],

    'colsample_bytree': [.6, .7, .8, .9, 1]

}



# Instantiate the regressor: gbm

gbm = XGBClassifier(n_estimators=10)



# Perform random search: grid_mse

xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 

                                    estimator = gbm, scoring = "accuracy", 

                                    verbose = 1, n_iter = 50, cv = 4)





# Fit randomized_mse to the data

xgb_random.fit(X_train, y_train)



# Print the best parameters and lowest RMSE

print("Best parameters found: ", xgb_random.best_params_)

print("Best accuracy found: ", xgb_random.best_score_)
model = XGBClassifier(n_estimators = xgb_random.best_params_['n_estimators'], learning_rate = xgb_random.best_params_['learning_rate'], max_depth = xgb_random.best_params_['max_depth'] , colsample_bytree = xgb_random.best_params_['colsample_bytree'] , random_state = 2020)

model.fit(X_train, y_train)

y_pred = model.predict(X_test).astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# Saving the result

pd.DataFrame({'Survived': y_pred}, index=test.index).reset_index().to_csv('Survived_pred_XGBoost.csv', index=False)

