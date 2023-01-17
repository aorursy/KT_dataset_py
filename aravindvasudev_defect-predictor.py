%matplotlib inline



import pickle

import pandas as pd



from IPython.display import HTML, display

import matplotlib



from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.model_selection import GridSearchCV, train_test_split



from sklearn.preprocessing import StandardScaler



pd.set_option('display.max_columns', None)

matplotlib.style.use('ggplot')
# reading the dataset

dataset_csv = pd.read_csv('./dataset/dataset.csv')

dataset_csv.head(10)
# `Low`, `Medium`, `High` to 1, 2, 3

dataset_csv['complexity'] = dataset_csv['complexity'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3)

dataset_csv['criticality'] = dataset_csv['criticality'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3)

dataset_csv['dependencies_criticality'] = dataset_csv['dependencies_criticality'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3)



# `Yes`, `No` to 1, 0

dataset_csv['BRD_Availability'] = dataset_csv['BRD_Availability'].apply(lambda x: 1 if x == 'Yes' else 0)

dataset_csv['FS_Availability'] = dataset_csv['FS_Availability'].apply(lambda x: 1 if x == 'Yes' else 0)

dataset_csv['change_in_schedule'] = dataset_csv['change_in_schedule'].apply(lambda x: 1 if x == 'Yes' else 0)

dataset_csv['requirement_validation'] = dataset_csv['requirement_validation'].apply(lambda x: 1 if x == 'Yes' else 0)

dataset_csv['automation_scope'] = dataset_csv['automation_scope'].apply(lambda x: 1 if x == 'Yes' else 0)

dataset_csv['environment_downtime'] = dataset_csv['environment_downtime'].apply(lambda x: 1 if x == 'Yes' else 0)



dataset_csv.head(10)
# Applying one-hot encoding

dataset_csv = pd.get_dummies(dataset_csv)

dataset_csv.head(10)
y = pd.DataFrame(dataset_csv['no_of_defects'])

y.head(10)
X = dataset_csv.drop('no_of_defects', 1)

X.head(10)
# A class structure to hold the dataset. Object of this will be dumped using Pickle

class Dataset:

    def __init__(self, X, y):

        self.X = X

        self.y = y
dataset = Dataset(X, y)

dataset.X.describe()
# Dumping the dataset object

with open('./dataset/dataset.pickle', 'wb') as saveFile:

    pickle.dump(dataset, saveFile)
# Split data into train and test set

X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.3, random_state=10)
Scaler = StandardScaler().fit(X_train)



X_train = Scaler.transform(X_train)
%time



# AdaBoostRegressor

param_grid_AdaBoostRegressor = {

        'learning_rate': [2, 1, 0.5, 0.01],

        'loss': ['linear', 'square', 'exponential']

    }



regr1 = GridSearchCV(AdaBoostRegressor(random_state=0), param_grid_AdaBoostRegressor)

regr1.fit(X_train, y_train.values.ravel())
%time



# RandomForestRegressor

regr2 = RandomForestRegressor(random_state=0)

regr2.fit(X_train, y_train.values.ravel())
%time



# GradientBoostingRegressor

regr3 = GradientBoostingRegressor(random_state=0)

regr3.fit(X_train, y_train.values.ravel())
# Scaling the test set

X_test = Scaler.transform(X_test)
# Computing r-squared values

table = """

<table>

<tr>

<th> Regressor </th>

<th> train set </th>

<th> test set </th>

</tr>

<tr>

<th> AdaBoost Regressor </th>

<td> {} </td>

<td> {} </td>

</tr>

<tr>

<th> RandomForest Regressor </th>

<td> {} </td>

<td> {} </td>

</tr>

<tr>

<th> GradientBoosting Regressor </th>

<td> {} </td>

<td> {} </td>

</tr>

</table>

"""



train_score_1 = regr1.score(X_train, y_train)

test_score_1  = regr1.score(X_test, y_test)



train_score_2 = regr2.score(X_train, y_train)

test_score_2  = regr2.score(X_test, y_test)



train_score_3 = regr3.score(X_train, y_train)

test_score_3  = regr3.score(X_test, y_test)



display(HTML(table.format(train_score_1, test_score_1, train_score_2, test_score_2, train_score_3, test_score_3)))
dataset_csv.plot.scatter(x='complexity', y='no_of_defects', s=dataset_csv['no_of_defects'])
dataset_csv.plot.scatter(x='no_of_requirements', y='no_of_defects', s=dataset_csv['no_of_defects'])
# Dumping the regressors

with open('./regressors/AdaBoostRegressor.pickle', 'wb') as f1, open('./regressors/RandomForestRegressor.pickle', 'wb') as f2, open('./regressors/GradientBoostingRegressor.pickle', 'wb') as f3:

    pickle.dump(regr1, f1)

    pickle.dump(regr2, f2)

    pickle.dump(regr3, f3)