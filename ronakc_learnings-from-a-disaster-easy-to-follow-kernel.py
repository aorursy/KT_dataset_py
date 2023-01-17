import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn import model_selection

from sklearn import linear_model, neighbors, ensemble, neural_network, svm



import os

import warnings

warnings.filterwarnings('ignore') # to avoid seeing warnings, you might wanna set it to 'once' or 'always'
data_train = pd.read_csv('/kaggle/input/titanic/train.csv',sep=',')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv',sep=',')
print("---- Data Size ----")

print("Training # Rows: ", data_train.shape[0], " | Columns: ", data_train.shape[1])

print("Test #     Rows: ", data_test.shape[0], " | Columns: ", data_test.shape[1])
data_train.head()
data_train.info()
data_test.info()
for col in ['PassengerId', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']:

    data_train[col] = data_train[col].astype('category')

    data_test[col] = data_test[col].astype('category')

    

#data_train['Survived'] = data_train['Survived'].astype('category')
data_train.describe(include='all')
data_test.describe(include='all')
# Let's see how Sex is correlated with Survival of people

g = sns.catplot(x="Sex", y="Survived", data=data_train,aspect=1, height=4, kind="bar", palette="Set2")

g.despine(left=True)

g.set_ylabels("Survival Probability")
# Let's see how Pclass (Class of travel) is correlated with Survival of people

g = sns.catplot(x="Pclass", y="Survived", data=data_train,aspect=1.5, height=4, kind="bar", palette="pastel")

g.despine(left=True)

g.set_ylabels("Survival Probability")
# Let's see how Embarked Port is correlated with Survival of people

g = sns.catplot(x="Embarked", y="Survived", data=data_train,aspect=1.5, height=4, kind="bar", palette="Set3")

g.despine(left=True)

g.set_ylabels("Survival Probability")
data_train['Age_bins'] = pd.cut(data_train['Age'], bins=[5,15,25,35,45,55,65,75,85,95,150])
g = sns.catplot(x="Age_bins", y="Survived",hue="Sex", data=data_train,aspect=2.7, height=5, kind="bar", palette="Set2")

g.set_ylabels("Survival Probability")
data_train['Fare_bins'] = pd.cut(data_train['Fare'], bins=[0,10,50,100,150,200,250,300,400,500,600])

g = sns.catplot(x="Fare_bins", y="Survived", hue="Sex", data=data_train,aspect=2.5, kind="bar", palette="Set3")

g.set_ylabels("Survival Probability")
sns.catplot(x="Survived", y="Fare", hue="Sex", kind="violin", col="Embarked", col_wrap=3,

               split=True, inner="quart", data=data_train, orient="v")
g = sns.catplot(x="SibSp", y="Survived", data=data_train, kind="bar", palette="Set2")

g.despine(left=True)

g.set_ylabels("Survival Probability")
g = sns.catplot(x="Parch", y="Survived", data=data_train, kind="bar", palette="Set3")

g.despine(left=True)

g.set_ylabels("Survival Probability")
data_train['Family_size'] = data_train['SibSp']+data_train['Parch'] + 1  # adding 1 to count the person currently considered too

g = sns.catplot(x="Family_size", y="Survived", data=data_train, kind="bar", palette="pastel")

g.despine(left=True)

g.set_ylabels("Survival Probability")
sns.catplot(y="Age", kind="violin", inner="box", data=data_train[data_train.Age >= 0])
sns.catplot(y="Family_size", kind="violin", inner="box", data=data_train[data_train.SibSp.notna()])
sns.catplot(x="Pclass", y="Survived", hue="Sex", kind="point", col="Embarked", col_wrap=3,

               data=data_train)
sns.catplot(x="Pclass", y="Fare", kind="violin", hue='Sex', split=True, col="Embarked", col_wrap=3, data=data_train)
# Name format is Lastname, Prefix. Firstname Middle Name ..

data_train['Name_Prefix'] = data_train['Name'].str.split(',', expand=True)[1].str.strip().str.split('.', expand=True)[0].str.lower()

data_train['Name_Prefix'].value_counts()
# re-assign some of the titles

data_train['Name_Prefix'] = data_train['Name_Prefix'].apply(lambda x: 'miss' if x in ['mlle','ms'] else x)

data_train['Name_Prefix'] = data_train['Name_Prefix'].apply(lambda x: 'mrs' if x in ['mme'] else x)



# replace rarely occuring prefixes

prefixes = (data_train['Name_Prefix'].value_counts() < 10)

data_train['Name_Prefix'] = data_train['Name_Prefix'].apply(lambda x: 'misc' if prefixes.loc[x] == True else x)

data_train['Name_Prefix'].value_counts()
g = sns.catplot(x="Name_Prefix", y="Survived", data=data_train, kind="bar", palette="Set3", aspect = 1.1)

g.despine(left=True)

g.set_ylabels("Survival Probability")
sns.catplot(x="Name_Prefix", y="Age", kind="box", data=data_train)
data_train.shape, data_test.shape
data_train = data_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked','Survived']]

data_all = data_train.append(data_test, ignore_index=True, sort=False)

data_all.shape
data_all[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].isna().sum()
data_all[data_all['Embarked'].isna()]
data_all.at[data_all[data_all['Embarked'].isna()].index,'Embarked'] = 'C'
data_all[data_all['Fare'].isna()]
fare = np.mean(data_all[(data_all['Pclass']==3) & (data_all['Embarked']=='S') & (data_all['Sex']=='male') & (data_all['SibSp']==0) & (data_all['Parch']==0) & (data_all['Age'] > 50) & (data_all['PassengerId'] != 1044)]['Fare'].values)
data_all.at[data_all[data_all['Fare'].isna()].index,'Fare'] = fare
data_all['Name_Prefix'] = data_all['Name'].str.split(',', expand=True)[1].str.strip().str.split('.', expand=True)[0].str.lower()

data_all['Name_Prefix'] = data_all['Name_Prefix'].apply(lambda x: 'miss' if x in ['mlle','ms'] else x)

data_all['Name_Prefix'] = data_all['Name_Prefix'].apply(lambda x: 'mrs' if x in ['mme'] else x)

# replace rarely occuring prefixes

prefixes = (data_all['Name_Prefix'].value_counts() < 10)

data_all['Name_Prefix'] = data_all['Name_Prefix'].apply(lambda x: 'misc' if prefixes.loc[x] == True else x)
for title in data_all['Name_Prefix'].unique():

    for sex in data_all['Sex'].unique():

        mean_age = np.mean(data_all[(~data_all['Age'].isna()) & (data_all['Sex'] == sex) & (data_all['Name_Prefix'] == title)]['Age'].values)

        data_all.at[data_all[(data_all['Age'].isna()) & (data_all['Sex'] == sex) & (data_all['Name_Prefix'] == title)].index,'Age'] = mean_age
data_all[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Name_Prefix']].isna().sum()
data_all['Family_size'] = data_all['SibSp']+data_all['Parch'] + 1
data_all = pd.get_dummies(data_all, columns=['Pclass','Sex','Embarked','Name_Prefix'])
age_scaler = MinMaxScaler()

data_all['Age'] = age_scaler.fit_transform(data_all['Age'].values.reshape(-1,1))

fare_scaler = MinMaxScaler()

data_all['Fare'] = fare_scaler.fit_transform(data_all['Fare'].values.reshape(-1,1))
data_all[['PassengerId','Age','Fare']].head(2)
data_train = data_all.iloc[:891]

data_test = data_all.iloc[891:]

data_train.shape, data_test.shape
columns_to_not_use = ['PassengerId', 'Survived', 'Ticket', 'Cabin', 'Name']
cv_splits = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7, random_state=42)
kwargs = {

    'logistic_regression':{'class_weight': None,  # default is None

                           'random_state': 42

                          },

    

    'random_forest': {'n_estimators': 10,              # default is 100

                     'max_depth': 6,                   # default is None

                     'max_features': 'auto',           

                     'max_leaf_nodes': 16,             # default is None

                     'class_weight': None,             # default is None

                     'criterion': 'entropy',           # default is gini

                     'oob_score': True,                # default is False

                     'random_state': 42

                     },

    

    'k_neighbours': {'n_neighbors': 10,                  # default is 5

                    'p': 2                               # default is 2 (euclidean_distance), p =1 (manhattan_distance)

                    },

    

    'mlp_classifier': {'hidden_layer_sizes': (16,8),      # default is (100,)

                      'activation': 'tanh',               # default is 'relu'

                      'solver': 'adam',                 

                      'max_iter': 150,                    # default is 200

                      'random_state': 42},

    

    'svm': {'C': 2,                       # default is 1

           'kernel':'rbf',                

           'random_state': 42}

}
algos = {

    'logistic_regression':linear_model.LogisticRegression(**kwargs['logistic_regression']),

    'random_forest':ensemble.RandomForestClassifier(**kwargs['random_forest']),

    'k_neighbours':neighbors.KNeighborsClassifier(**kwargs['k_neighbours']),

    'mlp_classifier':neural_network.MLPClassifier(**kwargs['mlp_classifier']),

    'svm':svm.SVC(**kwargs['svm'])

}
cv_results = {'Algorithm':[],                     # algorithm name

              'Mean Train Accuracy':[],           # Mean of training accuracy on all splits

              'Mean Test Accuracy':[],            # Mean of test accuracy on all splits

              'Test Standard deviation':[],       # Standard deviation of test accuracy on all splits 

                                                  # (this is to know how worse the algorithm can perform)

              'Fit Time': []}                     # how fast the algorithm converges
for alg_name,alg in algos.items():

    cv_results['Algorithm'].append(alg_name)

    

    cross_val = model_selection.cross_validate(alg, 

                                               data_train.loc[:, ~data_train.columns.isin(columns_to_not_use)], 

                                               data_train['Survived'],

                                               cv  = cv_splits,

                                               return_train_score=True,

                                               return_estimator=False

                                              )

    

    cv_results['Mean Train Accuracy'].append(cross_val['train_score'].mean())

    cv_results['Mean Test Accuracy'].append(cross_val['test_score'].mean())

    cv_results['Test Standard deviation'].append(cross_val['test_score'].std()*3)

    cv_results['Fit Time'].append(cross_val['fit_time'].mean())

    
cv_results_df = pd.DataFrame.from_dict(cv_results)

cv_results_df.sort_values(by=['Mean Test Accuracy'], inplace=True, ascending=False)

cv_results_df
# store the predictions in a dictionary

y_predicted = {}
for alg_name,alg in algos.items():

    

    alg.fit(data_train.loc[:, ~data_train.columns.isin(columns_to_not_use)], data_train['Survived'])

    y_predicted[alg_name] = alg.predict(data_test.loc[:, ~data_test.columns.isin(columns_to_not_use)])

    
# create a dataframe and write to a csv file

for alg_name in algos.keys():

    results_dict = {'PassengerId':data_test['PassengerId'].values.tolist(), 'Survived':list(map(int, y_predicted[alg_name]))}

    results_df = pd.DataFrame.from_dict(results_dict)

    results_df.to_csv(alg_name+'.csv', index=False)