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
import pandas as pd

import numpy as np

import math as mt

from sklearn.impute import SimpleImputer

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier





pd.set_option('display.max_columns', None)

pd.set_option('display.width', 640)

pd.set_option('display.float_format', lambda x: '%.3f' % x)



# Import the Titanic data:

data = pd.read_csv('/kaggle/input/titanic/train.csv', sep=',', header=0)



# Since a vast majority of the Cabin values are missing and it is not possible to make an educated guess for the missing Cabin values by using the other Cabin values,

# Cabin feature is dropped:

data.drop(columns='Cabin', inplace=True)





# A new feature called 'Title' is created using names of the passengers:

data['Name'] = data['Name'].apply(lambda x: x.replace(' ', ''))

data['Title'] = data['Name'].apply(lambda x: x[(x.index(',')+1):x.index('.')])





# There are 177 null Age values out of 891 values. Passengers' 'Age's which are null are assigned value by taking average 'Age's of

# the passengers who have the same Title as the one with null Age value:

data_1 = data.copy(deep=True)

# data_1.info()

data_1['Age'] = data_1.groupby(by='Title')['Age'].transform(lambda x: x.fillna(x.mean()))





# Since frequencies of "Title"s other than "Mr", "Miss", "Mrs" and "Master" are very low, it is useless to make model try to find a correlation

# between those Title values and the related Survived values. Therefore  Title values other than "Mr", "Miss", "Mrs" and "Master"

# are replaced by "Other":

"""

data_1['Title'].value_counts()

Mr             517

Miss           182

Mrs            125

Master          40

Dr               7

Rev              6

Major            2

Col              2

Mlle             2

Sir              1

Ms               1

Don              1

Lady             1

theCountess      1

Mme              1

Jonkheer         1

Capt             1

"""

data_1.loc[~data_1['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Other'





# The observations (2-many) with null Embarked value are dropped:

data_1.dropna(subset=['Embarked'], inplace=True)



# There is no missing feature value now.



# Number of beloved ones of passengers and based on this, whether person is alone is determined:

data_1['Num_of_beloved'] = data_1['SibSp'] + data_1['Parch']

data_1['Is_alone'] = data_1['Num_of_beloved'].apply(lambda x: 0 if x>0 else 1)



# Females' survival prob. is higher than males:

sns.catplot(x='Sex', y='Survived', row='Embarked', col='Pclass', data=data_1, kind='bar')



# Passengers with Pclass=1 have higher survival prob:

sns.catplot(x='Pclass', y='Survived', row='Embarked', col='Sex', data=data_1, kind='bar').col_names



# Two dataframes, one for male and the other for female, are created:

women = data_1[data_1['Sex'] == 'female']

men = data_1[data_1['Sex'] == 'male']



# Below, survival rates of men wrt their ages are calculated and the survival graph is sketched:

plt.clf()

count, edges, plot = plt.hist([men.Age, men[men['Survived'] == 1].Age], bins=np.linspace(0, 80, 11), label=['All the men','The Survivors'])

plt.legend(loc='upper right')

plt.show()

men_survival = {'Age interval': np.linspace(0, 80, 11)[1:], 'Survival rate': count[1]/count[0]}

men_survival_rates = pd.DataFrame(men_survival)

men_survival_rates.sort_values(by='Age interval', ascending=True, inplace=True)





# Below, survival rates of women wrt their ages are calculated and the survival graph is sketched:

plt.clf()

count, edges, plot = plt.hist([women.Age, women[women['Survived'] == 1].Age], bins=np.linspace(0, 80, 11), label=['All the women','The Survivors'])

plt.legend(loc='upper right')

plt.show()

survival_rate = (count[1][0:8]/count[0][0:8]).tolist() + [1, 1]

women_survival = {'Age interval': np.linspace(0, 80, 11)[1:], 'Survival rate': survival_rate}

women_survival_rates = pd.DataFrame(women_survival)

women_survival_rates.sort_values(by='Age interval', ascending=True, inplace=True)





# Based on the results of Age&Survival relations found above, a new feature called "Age_Group" is created using "Age" values of passengers below:

for i in data_1.index.to_list():

    if data_1.loc[i, 'Age'] <= 8:

        data_1.loc[i, 'Age_Group'] = 1

    elif data_1.loc[i, 'Age'] <= 16:

        data_1.loc[i, 'Age_Group'] = 2

    elif data_1.loc[i, 'Age'] <= 24:

        data_1.loc[i, 'Age_Group'] = 3

    elif data_1.loc[i, 'Age'] <= 32:

        data_1.loc[i, 'Age_Group'] = 4

    elif data_1.loc[i, 'Age'] <= 40:

        data_1.loc[i, 'Age_Group'] = 5

    elif data_1.loc[i, 'Age'] <= 48:

        data_1.loc[i, 'Age_Group'] = 6

    elif data_1.loc[i, 'Age'] <= 64:

        data_1.loc[i, 'Age_Group'] = 7

    else:

        data_1.loc[i, 'Age_Group'] = 8



data_1['Age_Group'] = data_1['Age_Group'].astype(int)







# Below, survival rates of men wrt their "Fare"s are calculated and the survival graph is sketched:

plt.clf()

count, edges, plot = plt.hist([men.Fare, men[men['Survived'] == 1].Fare], bins=np.linspace(0, 512, 21), label=['All the men','The Survivors'])

plt.legend(loc='upper right')

plt.show()

men_survival = {'Age interval': np.linspace(0, 512, 21)[1:], 'Survival rate': count[1]/count[0]}

men_survival_rates = pd.DataFrame(men_survival)

men_survival_rates.sort_values(by='Age interval', ascending=True, inplace=True)





# Below, survival rates of women wrt their "Fare"s are calculated and the survival graph is sketched:

plt.clf()

count, edges, plot = plt.hist([women.Fare, women[women['Survived'] == 1].Fare], bins=np.linspace(0, 512, 21), label=['All the women','The Survivors'])

plt.legend(loc='upper right')

plt.show()

survival_rate = (count[1][0:8]/count[0][0:8]).tolist() + [1, 1]

women_survival = {'Age interval': np.linspace(0, 80, 11)[1:], 'Survival rate': survival_rate}

women_survival_rates = pd.DataFrame(women_survival)

women_survival_rates.sort_values(by='Age interval', ascending=True, inplace=True)







# Based on the results of Fare&Survival relations found above, a new feature called "Fare_Group" is created using "Fare" values of passengers below:

for ii in data_1.index.to_list():

    if data_1.loc[ii, 'Fare'] <= 25.6:

        data_1.loc[ii, 'Fare_Group'] = 1

    elif data_1.loc[ii, 'Fare'] <= 51.2:

        data_1.loc[ii, 'Fare_Group'] = 2

    elif data_1.loc[ii, 'Fare'] <= 76.8:

        data_1.loc[ii, 'Fare_Group'] = 3

    elif data_1.loc[ii, 'Fare'] <= 102.4:

        data_1.loc[ii, 'Fare_Group'] = 4

    elif data_1.loc[ii, 'Fare'] <= 128:

        data_1.loc[ii, 'Fare_Group'] = 5

    elif data_1.loc[ii, 'Fare'] <= 153.6:

        data_1.loc[ii, 'Fare_Group'] = 6

    else:

        data_1.loc[ii, 'Fare_Group'] = 7



data_1['Fare_Group'] = data_1['Fare_Group'].astype(int)



# Below, features which won't be used in model training are dropped. Also, multiple features with numerical values are created for each of a categorical feature.

# And the redundant newly created featrues are dropped, too:

data_2 = data_1.copy(deep=True)

data_2.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

data_2 = pd.get_dummies(data=data_2)

data_2.drop(columns=['Sex_male', 'Embarked_S', 'Title_Other'], inplace=True)



# First, KNN will be tried to be fitted on the data. Therefore, all the features should be on the same scale. Below, that operation is performed:

data_3 = data_2.copy(deep=True)

scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(data_3[data_3.columns.to_list()[1:]])

data_3[data_3.columns.to_list()[1:]] = scaler.transform(data_3[data_3.columns.to_list()[1:]])

data_3.reset_index(drop=True, inplace=True)



# Below, data set is splitted into train and test sets:

x_train, x_test, y_train, y_test = train_test_split(data_3[data_3.columns.to_list()[1:]], data_3['Survived'], test_size=0.2, random_state=42, stratify=data_3['Survived'])



# Below, KNN is tried:

knn = KNeighborsClassifier()

parameter_grid = {'n_neighbors': np.arange(15)[2:]}

grsearch = GridSearchCV(estimator=knn, param_grid=parameter_grid, scoring='accuracy', n_jobs=-1, cv=5)

grsearch.fit(x_train, y_train)

grsearch.best_score_ # 0.8270042194092827

grsearch.best_params_ # {'n_neighbors': 5}





knn_1 = KNeighborsClassifier(n_neighbors=5)

knn_1.fit(x_train, y_train)

y_pred = knn_1.predict(x_test)

accuracy_score(y_test, y_pred) # 0.797752808988764



# Below, LogisticRegression is tried:

logreg = LogisticRegression(random_state=42, n_jobs=-1)

C_param_range = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

grsearch_1 = GridSearchCV(estimator=logreg, param_grid=C_param_range, scoring='accuracy', n_jobs=-1, cv=5)

grsearch_1.fit(x_train, y_train)

grsearch_1.best_score_ # 0.8255977496483825

grsearch_1.best_params_ # {'C': 1}



logreg_1 = LogisticRegression(random_state=42, n_jobs=-1, C=1)

logreg_1.fit(x_train, y_train)

pred= logreg_1.predict(x_test)

accuracy_score(y_test, pred) # 0.8146067415730337



# Below, DecisionTreeClassifier is tried:

tree = DecisionTreeClassifier(random_state=42)

parameter_grid = {'max_depth': np.arange(8)[2:], 'min_samples_leaf': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4]}

grsearch_2 = GridSearchCV(estimator=tree, param_grid=parameter_grid, scoring='accuracy', n_jobs=-1, cv=5)

grsearch_2.fit(x_train, y_train)

grsearch_2.best_score_ # 0.8227848101265823

grsearch_2.best_params_ # {'max_depth': 3, 'min_samples_leaf': 0.04}



tree_1 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.04, random_state=42)

tree_1.fit(x_train, y_train)

pred_y = tree_1.predict(x_test)

accuracy_score(y_test, pred_y) # 0.8033707865168539



# Below, VotingClassifier using KNN, LogisticRegression and DecisionTreeClassifier for ultimate prediction is trie:

knn_2 = KNeighborsClassifier(n_neighbors=5)

logreg_2 = LogisticRegression(random_state=42, C=1)

tree_2 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.04, random_state=42)

vote = VotingClassifier(estimators=[('knn', knn_2), ('logreg', logreg_2), ('tree', tree_2)], voting='hard', n_jobs=-1)

vote.fit(x_train, y_train)

pred_y = vote.predict(x_test)

accuracy_score(y_test, pred_y) # 0.8202247191011236





# Below, RandomForestClassifier is tried:

parameter_grid = {'n_estimators': [50, 100, 150], 'max_depth': np.arange(7)[1:], 'max_features': np.linspace(0, 1, 11)[1:]}

rforest = RandomForestClassifier(random_state=42)

grsearch_3 = GridSearchCV(estimator=rforest, param_grid=parameter_grid, scoring='accuracy', n_jobs=-1, cv=5)

grsearch_3.fit(x_train, y_train)

grsearch_3.best_score_ # 0.8368495077355836

grsearch_3.best_params_ # {'max_depth': 4, 'max_features': 0.7000000000000001, 'n_estimators': 150}





rforest_1 = RandomForestClassifier(max_depth=4, max_features=0.7, n_estimators=150, random_state=42, n_jobs=-1)

rforest_1.fit(x_train, y_train)

y_pred = rforest_1.predict(x_test)

accuracy_score(y_test, y_pred) # 0.8146067415730337



# Below, BaggingClassifier is tried:

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.04, random_state=42)

parameter_grid = {'n_estimators': [50, 100, 150]}

bag = BaggingClassifier(base_estimator=tree, random_state=42)

grsearch_4 = GridSearchCV(bag, param_grid=parameter_grid, scoring='accuracy', n_jobs=-1, cv=5)

grsearch_4.fit(x_train, y_train)

grsearch_4.best_score_ # 0.810126582278481

grsearch_4.best_params_ #{'n_estimators': 50}



bag_1 = BaggingClassifier(base_estimator=tree, n_estimators=50, random_state=42, n_jobs=-1)

bag_1.fit(x_train, y_train)

y_pred = bag_1.predict(x_test)

accuracy_score(y_test, y_pred) # 0.7865168539325843





# Below, AdaBoostClassifier is tried:

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.04, random_state=42)

parameter_grid = {'n_estimators': [50, 100, 150], 'learning_rate': np.linspace(0, 1, 11)[1:]}

adaptiveb = AdaBoostClassifier(base_estimator=tree, random_state=42)

grsearch_5 = GridSearchCV(estimator=adaptiveb, param_grid=parameter_grid, scoring='accuracy', n_jobs=-1,cv=5)

grsearch_5.fit(x_train, y_train)

grsearch_5.best_score_ # 0.8270042194092827

grsearch_5.best_params_ # {'learning_rate': 0.5, 'n_estimators': 50}



adaptiveb_1 = AdaBoostClassifier(base_estimator=tree, n_estimators=50, learning_rate=0.5, random_state=42)

adaptiveb_1.fit(x_train, y_train)

y_pred = adaptiveb_1.predict(x_test)

accuracy_score(y_test, y_pred) # 0.8033707865168539





# Below, GradientBoostingClassifier is tried:

parameter_grid = {'learning_rate': np.linspace(0, 1, 11)[1:], 'n_estimators': [50, 100, 150], 'max_depth': np.arange(8)[2:], 'max_features': np.linspace(0, 1, 11)[1:]}

gradientb = GradientBoostingClassifier(random_state=42)

grsearch_6 = GridSearchCV(estimator=gradientb, param_grid=parameter_grid, scoring='accuracy', n_jobs=-1, cv=5)

grsearch_6.fit(x_train, y_train)

grsearch_6.best_score_ # 0.8368495077355836

grsearch_6.best_params_ # {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 0.8, 'n_estimators': 50}





gradientb_1 = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.8, n_estimators=50, random_state=42)

gradientb_1.fit(x_train, y_train)

y_pred = gradientb_1.predict(x_test)

accuracy_score(y_test, y_pred) # 0.8539325842696629



# RandomForestClassifier and GradientBoostingClassifier have the best score among all the classifiers tried so far.

# GradientBoostingClassifier is a little bit better than RandomForestClassifier.

# So, GradientBoostingClassifier will be used in predictions:





# Below, train data and the test data which requires survival predictions for the passengers are imported:



train_data = pd.read_csv('/kaggle/input/titanic/train.csv', sep=',', header=0)

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', sep=',', header=0)

# To be able to concatenate train and test data, 'Survived' column is added to the test data, but it will be dropped later. No problem:

test_data['Survived'] = 0

test_data = test_data.copy(deep=True)[train_data.columns.to_list()]



# Train and test data is combined:

data = pd.concat([train_data, test_data], ignore_index=True)



# Since a vast majority of the Cabin values are missing and it is not possible to make an educated guess for the missing Cabin values by using the other Cabin values,

# Cabin feature is dropped:

data.drop(columns='Cabin', inplace=True)





# A new feature called 'Title' is created using names of the passengers:

data['Name'] = data['Name'].apply(lambda x: x.replace(' ', ''))

data['Title'] = data['Name'].apply(lambda x: x[(x.index(',')+1):x.index('.')])





# There are 177 null Age values out of 891 values. Passengers' 'Age's which are null are assigned value by taking average 'Age's of

# the passengers who have the same Title as the one with null Age value:

data_1 = data.copy(deep=True)

# data_1.info()

data_1['Age'] = data_1.groupby(by='Title')['Age'].transform(lambda x: x.fillna(x.mean()))



# Title of passengers is arranged below:

data_1.loc[~data_1['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Other'





# The observations (2-many) with null Embarked value are dropped:

data_1.dropna(subset=['Embarked'], inplace=True)



# There is no missing feature value now.



# Number of beloved ones of passengers and based on this, whether person is alone is determined:

data_1['Num_of_beloved'] = data_1['SibSp'] + data_1['Parch']

data_1['Is_alone'] = data_1['Num_of_beloved'].apply(lambda x: 0 if x>0 else 1)





# Based on the results of Age&Survival relations found, a new feature called "Age_Group" is created using "Age" values of passengers below:

for i in data_1.index.to_list():

    if data_1.loc[i, 'Age'] <= 8:

        data_1.loc[i, 'Age_Group'] = 1

    elif data_1.loc[i, 'Age'] <= 16:

        data_1.loc[i, 'Age_Group'] = 2

    elif data_1.loc[i, 'Age'] <= 24:

        data_1.loc[i, 'Age_Group'] = 3

    elif data_1.loc[i, 'Age'] <= 32:

        data_1.loc[i, 'Age_Group'] = 4

    elif data_1.loc[i, 'Age'] <= 40:

        data_1.loc[i, 'Age_Group'] = 5

    elif data_1.loc[i, 'Age'] <= 48:

        data_1.loc[i, 'Age_Group'] = 6

    elif data_1.loc[i, 'Age'] <= 64:

        data_1.loc[i, 'Age_Group'] = 7

    else:

        data_1.loc[i, 'Age_Group'] = 8



data_1['Age_Group'] = data_1['Age_Group'].astype(int)







# Based on the results of Fare&Survival relations found, a new feature called "Fare_Group" is created using "Fare" values of passengers below:

for ii in data_1.index.to_list():

    if data_1.loc[ii, 'Fare'] <= 25.6:

        data_1.loc[ii, 'Fare_Group'] = 1

    elif data_1.loc[ii, 'Fare'] <= 51.2:

        data_1.loc[ii, 'Fare_Group'] = 2

    elif data_1.loc[ii, 'Fare'] <= 76.8:

        data_1.loc[ii, 'Fare_Group'] = 3

    elif data_1.loc[ii, 'Fare'] <= 102.4:

        data_1.loc[ii, 'Fare_Group'] = 4

    elif data_1.loc[ii, 'Fare'] <= 128:

        data_1.loc[ii, 'Fare_Group'] = 5

    elif data_1.loc[ii, 'Fare'] <= 153.6:

        data_1.loc[ii, 'Fare_Group'] = 6

    else:

        data_1.loc[ii, 'Fare_Group'] = 7



data_1['Fare_Group'] = data_1['Fare_Group'].astype(int)



# It is guaranteed that there is not any row dropped from test data:

data_1.loc[data_1['PassengerId'] >= 892, 'PassengerId'].count() # 418 many passengers available in test data

test_data_2 = data_1.loc[data_1['PassengerId'] >= 892]

train_data_2 = data_1.loc[data_1['PassengerId'] < 892]







# Below, features which won't be used in model training are dropped. Also, multiple features with numerical values are created for each of a categorical feature.

# And the redundant newly created featrues are dropped, too:

train_data_2 = train_data_2.copy(deep=True)

train_data_2.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

train_data_2 = pd.get_dummies(data=train_data_2)

train_data_2.drop(columns=['Sex_male', 'Embarked_S', 'Title_Other'], inplace=True)





# Below, features which won't be used in prediction are dropped:

test_data_2 = test_data_2.copy(deep=True)

test_data_2.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

test_data_2 = pd.get_dummies(data=test_data_2)

test_data_2.drop(columns=['Sex_male', 'Embarked_S', 'Title_Other'], inplace=True)



# There is a single row whose 'Fare' value is missing within test_date_2. A 'Fare' value is assigned to that record as follows:

test_data_2['Fare'] = test_data_2.groupby(by='Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))





# Below, GradientBoostingClassifier is used to fit to the training data. Parameters used in fitting are determined in the hyperparameter tuning

# performed so far:

gradientb_2 = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.8, n_estimators=50, random_state=42)

gradientb_2.fit(train_data_2[train_data_2.columns.to_list()[1:]], train_data_2['Survived'])

y_pred = gradientb_2.predict(test_data_2[test_data_2.columns.to_list()[1:]])





# Predictions are formtatted based on the specified upload format of the file:

output = pd.DataFrame({'PassengerId': data_1.loc[data_1['PassengerId'] >= 892, 'PassengerId'].values.tolist(), 'Survived': y_pred})

output.to_csv('/kaggle/working/Titanic_Predictions.csv', sep=',', index=False, header=True)





import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")