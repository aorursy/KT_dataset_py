# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import riiideducation

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn import model_selection

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



dtype={

        'row_id': 'int64', 

        'timestamp': 'int64', 

        'user_id': 'int32', 

        'content_id': 'int16', 

        'content_type_id': 'int8',

        'task_container_id': 'int16', 

        'user_answer': 'int8', 

        'answered_correctly': 'int8', 

        'prior_question_elapsed_time': 'float32', 

        'prior_question_had_explanation': 'boolean'

    } 

train = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', dtype = dtype, low_memory = False, nrows= 10**6)

train.head()
def dataframe_information(dataframe):

    

    # Shape

    print(dataframe.shape)

    

    # Information Values

    Information_df = pd.DataFrame()

    

    feature_list = []

    count_list = []

    percent_list = []

    unique_value_list = []

    mean_list = []

    median_list = []

    mode_list = []

    

    for col in dataframe.columns:

        

        feature_list.append(col)

        count_list.append(dataframe[col].isnull().sum())

        percent_list.append(dataframe[col].isnull().sum() * 100/len(dataframe))

        unique_value_list.append(dataframe[col].nunique())

        mean_list.append(int(dataframe[col].mean()))

        median_list.append(int(dataframe[col].median()))

        mode_list.append(int(dataframe[col].mode()[0]))

        



    Information_df['feature'] = feature_list

    Information_df['Number of missing values'] = count_list

    Information_df['Percentage_of_missing'] = percent_list

    Information_df['Unique_values_for_each_column'] = unique_value_list

    Information_df['Mean for the variable'] = mean_list

    Information_df['Median for the variable'] = median_list

    Information_df['Mode for the variable'] = mode_list

    

    return Information_df



dataframe_information(train)
features = ['row_id','timestamp', 'user_id', 'content_id','content_type_id', 'task_container_id', 'prior_question_elapsed_time',

       'prior_question_had_explanation']



target = ['answered_correctly']

def preprocessing(dataframe, features, target):

    

    # Basic data information

    print(dataframe.shape)

    

    # Filling Null values

    dataframe = dataframe.fillna(0)

    

    # Change the categorical to numerical

    mapping = {False: 0, True: 1}

    dataframe.loc[:, 'prior_question_had_explanation'] = dataframe.loc[:, 'prior_question_had_explanation'].map(mapping)

    

    # Dropping Unnecessary columns

    decided_col_list = ['row_id','timestamp', 'user_id', 'content_id','content_type_id', 'task_container_id','user_answer', 'answered_correctly', 'prior_question_elapsed_time',

       'prior_question_had_explanation']





    for col in dataframe.columns:

        if str(col) not in decided_col_list:

            dataframe = dataframe.drop(col, axis = 1)

    

    # Missing Values and infinite values replacement    

    dataframe.loc[:, 'prior_question_elapsed_time'] = dataframe.loc[:, 'prior_question_elapsed_time'].replace(np.nan, dataframe['prior_question_elapsed_time'].mode()[0])

    dataframe.loc[:, 'prior_question_had_explanation'] = dataframe.loc[:, 'prior_question_had_explanation'].replace(np.nan, dataframe['prior_question_had_explanation'].median())

    

    # Standardize the values:

    features_tobe_standardized = ['timestamp', 'prior_question_elapsed_time']

    dataframe.timestamp = (dataframe.timestamp - dataframe.timestamp.mean()) / dataframe.timestamp.std()

    dataframe.prior_question_elapsed_time = (dataframe.prior_question_elapsed_time - dataframe.prior_question_elapsed_time.mean()) / dataframe.prior_question_elapsed_time.std()            

    

    

    # Removing answered_correctly which is equal to -1

    #dataframe = dataframe[dataframe['answered_correctly'] != -1]

    

    # Groupby user id

    #dataframe = dataframe.groupby('user_id')['answered_correctly'].sum()

    #dataframe['intelligence'] = dataframe

    return dataframe, dataframe[features]



train = preprocessing(train, features, target)[0]

train.head()
train.corr()
plt.hist(train['answered_correctly'])

plt.show()
train.corr()
from sklearn.model_selection import cross_val_score



def cross_validation(model, X, y, num_folds):

    kfolds = model_selection.KFold(n_splits = num_folds)

    results = cross_val_score(model, X, y, cv = kfolds, scoring = 'neg_mean_squared_error')

    print(f'Mean: {results.mean()}, Standard Deviation: {results.std()}')

    return [results.mean(), results.std()]
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import roc_auc_score



models = [LinearRegression(),Ridge(), Lasso(), ElasticNet(), DecisionTreeRegressor(), SGDClassifier(loss = 'log')]

regressor = models[4]

regressor.fit(train[features], train[target])
# # Spot check the algorithms

# for model in models:

#     cross_validation(model, train[features], train[target], 7)
try:

    env = riiideducation.make_env()

except:

    pass

iter_test = env.iter_test()



for (test_df, sample_prediction) in iter_test:

    X_test = preprocessing(test_df, features, target)[1]

    test_df['answered_correctly'] = regressor.predict(X_test)

    test_df = test_df.fillna(0)

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    print(test_df.head(5))
test_df