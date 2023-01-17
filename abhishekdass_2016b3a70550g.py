import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Extract training data 

data_train = pd.read_csv("../input/bits-f464-l1/train.csv") 

data_train = data_train.drop(["id"], axis = 1)

time_train = data_train.loc[ : , "time"]  



# Extract test data

data_test = pd.read_csv("../input/bits-f464-l1/test.csv") 

ids = data_test.loc[ : , "id"]  

data_test = data_test.drop(["id"], axis = 1)

time_test = data_test.loc[ : , "time"]
data_train.head()
data_test.head()
data_train.describe()
data_test.describe()
# Number of rows with atleast one NaN value in the train set



print("Count of rows with NaN values in train set: " + str(data_train.isnull().any(axis=1).sum()))
# Number of rows with atleast one NaN value in the test sets



print("Count of rows with NaN values in test set: " + str(data_test.isnull().any(axis=1).sum()))
# Split train set for each individual

agent_data_train = {}



for i in range(0, 7):

    agent_data_train[i] = data_train[data_train['a' + str(i)] == 1].drop(labels=['a0','a1','a2','a3','a4','a5','a6'], axis=1)
# Split test set for each individual

agent_data_test = {}



for i in range(0, 7):

    agent_data_test[i] = data_test[data_test['a' + str(i)] == 1].drop(labels=['a0','a1','a2','a3','a4','a5','a6'], axis=1)
# Spilitting the features and output variable for the train set

agent_data_train_x = {}

agent_data_train_y = {}



for i in range(0, 7):

    agent_data_train_x[i] = agent_data_train[i].drop(["label"], axis = 1)

    agent_data_train_x[i].set_index("time", inplace = True, drop = False)

    agent_data_train_y[i] = pd.DataFrame(agent_data_train[i].loc[ : , ["time", "label"]])

    agent_data_train_y[i].set_index("time", inplace = True)
# Spilitting the features and output variable for the test set

agent_data_test_x = {}



for i in range(0, 7):

    agent_data_test_x[i] = agent_data_test[i]

    agent_data_test_x[i].set_index("time", inplace = True, drop = False)
agent_data_train_x[0].head()
agent_data_train_y[0].head()
agent_data_test_x[0].head()
# # Standardise the train set



# from sklearn.preprocessing import StandardScaler



# for i in range(0, 7):

#     agent_data_train_x[i] = pd.DataFrame(StandardScaler().fit_transform(agent_data_train_x[i]), columns = agent_data_train_x[i].columns)

#     #agent_data_train_y[i] = pd.DataFrame(StandardScaler().fit_transform(agent_data_train_y[i]), columns = agent_data_train_y[i].columns)
# # Standardise the test set



# from sklearn.preprocessing import StandardScaler



# for i in range(0, 7):

#     agent_data_test_x[i] = pd.DataFrame(StandardScaler().fit_transform(agent_data_test_x[i]), columns = agent_data_test_x[i].columns)
agent_data_train_x[6].head()
agent_data_test_x[6].head()
agent_data_train_x_FS = {}

agent_data_test_x_FS = {}



for k in range(0, 7):

    corr = agent_data_train_x[k].corr()

    columns = np.full((corr.shape[0],), True, dtype=bool)

    

    for i in range(corr.shape[0]):

        for j in range(i+1, corr.shape[0]):

            if abs(corr.iloc[i,j]) >= 0.99:

                if columns[j]:

                    columns[j] = False

    

    selected_columns = agent_data_train_x[k].columns[columns]

    print(selected_columns.shape)

    agent_data_train_x_FS[k] = agent_data_train_x[k][selected_columns]

    agent_data_test_x_FS[k] = agent_data_test_x[k][selected_columns]
agent_data_train_x = agent_data_train_x_FS

agent_data_test_x = agent_data_test_x_FS
# # Remove Zero Var variables 

# for i in range(0, 7):

#     agent_data_train_x[i].drop(agent_data_train_x[i].std()[agent_data_train_x[i].std() < 0.05].index.values, axis = 1, inplace=True)

    
# selected_columns = agent_data_train_x[2].columns

# agent_data_test_x[2][selected_columns].shape
# agent_data_test_x_FS_V = {}



# for i in range(0, 7):

#     selected_columns = agent_data_train_x[i].columns

#     agent_data_test_x_FS_V[i] = agent_data_test_x[i][selected_columns]
# agent_data_test_x = agent_data_test_x_FS_V
# from sklearn.decomposition import PCA



# train_x = {}

# train_y = {}

# # pca - keep 90% of variance

# for i in range(0, 7):

#     pca = PCA(.98)

#     train_x[i] = pd.DataFrame(pca.fit_transform(agent_data_train_x[i]))

#     train_y[i] = agent_data_train_y[i]

#     print(train_x[i].shape)

#     print()
# from sklearn.preprocessing import StandardScaler



# for i in range(0, 7):

#     train_x[i] = pd.DataFrame(StandardScaler().fit_transform(train_x[i]), columns = train_x[i].columns)
# For Bypassing the PCA

train_x = {}

train_y = {}



for i in range(0, 7):

    train_x[i] = agent_data_train_x[i]

    train_y[i] = agent_data_train_y[i]



test_x = {}



for i in range(0, 7):

    test_x[i] = agent_data_test_x[i]
split_index = int(0.8 * len(train_x[0]))



train_x_train = {}

train_y_train = {}

train_x_test = {}

train_y_test = {}



for i in range(0, 7):

    train_x_train[i] = train_x[i][:split_index]

    train_x_test[i] = train_x[i][split_index:]

    train_y_train[i] = train_y[i][:split_index]

    train_y_test[i] = train_y[i][split_index:]
train_x_train[0].head()
train_y_train[0].head()
train_x_test[0].head()
train_y_test[0].head()
test_x[0].head()
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor, AdaBoostClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn.svm import SVR

from sklearn.neural_network import MLPClassifier,MLPRegressor

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from sklearn.metrics import mean_squared_error
for i in range(0, 7):

    etr = ExtraTreesRegressor(n_estimators = 200, random_state=120, n_jobs=-1)

    etr.fit(train_x_train[i], np.ravel(train_y_train[i]))

    print(np.sqrt(mean_squared_error(np.ravel(train_y_test[i]), etr.predict(train_x_test[i]))))
for i in range(0, 7):

    regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60, bootstrap=True)

    regressor.fit(train_x_train[i], np.ravel(train_y_train[i]))

    print(np.sqrt(mean_squared_error(np.ravel(train_y_test[i]), regressor.predict(train_x_test[i]))))
# regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60, bootstrap=True)

# regressor.fit(train_x_train[0], np.ravel(train_y_train[0]))



# regressor.feature_importances_
# for i in range(0, 7):

#     print("Agent # " + str(i) + ": ")

#     for k in ['rbf']:

#         svr = SVR(kernel=k)

#         svr.fit(train_x_train[i], np.ravel(train_y_train[i]))

#         print(np.sqrt(mean_squared_error(np.ravel(train_y_test[i]), svr.predict(train_x_test[i]))))
# We see good resuls only with linear and rbf
predicted_y = {}



for i in range(0, 7):

    regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

    regressor.fit(train_x[i], np.ravel(train_y[i]))

    predicted_y[i] = regressor.predict(test_x[i])
length = len(predicted_y[0])

predicted_y[0].shape
result = pd.DataFrame(columns=('id', 'label'))

k = 0

for i in range(0, length):

    for j in range(0, 7):

        result.loc[k] = [int(ids[k]), predicted_y[j][i]]

        k = k + 1
result["id"] = result["id"].astype(int)

result.info()
result.to_csv('submission_3.csv', header=True, index=False)