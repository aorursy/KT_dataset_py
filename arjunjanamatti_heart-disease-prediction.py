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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
raw_data = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')

raw_data
print('Number of null values in each column')

print(raw_data.isnull().sum())



print()

columns_with_null_values = []

for i in range(len(raw_data.columns)):

    if raw_data[raw_data.columns[i]].isnull().sum() > 0:

        columns_with_null_values.append(raw_data.columns[i])

print('Columns with null values: ', columns_with_null_values)
raw_data.info()
for i in range(len(columns_with_null_values)):

    print(columns_with_null_values[i], raw_data[columns_with_null_values[i]].unique())
for i in range(len(columns_with_null_values)):

    a = len(raw_data[columns_with_null_values[i]].unique())

    if a <= 5:

        raw_data[columns_with_null_values[i]].fillna(raw_data[columns_with_null_values[i]].mode()[0], inplace=True)

raw_data
for i in range(len(columns_with_null_values)):

    a = len(raw_data[columns_with_null_values[i]].unique())

    mean_value = raw_data[columns_with_null_values[i]].mean()

    median_value = raw_data[columns_with_null_values[i]].median()

    if a > 10:

        if abs(mean_value - median_value) < 100:

#             print("fill with mean values for column: ",listed_columns_numeric_1[i] )

            raw_data[columns_with_null_values[i]] = raw_data[columns_with_null_values[i]].fillna(mean_value)

        else:

#             print("fill with median values for column: ", listed_columns_numeric_1[i])

            raw_data[columns_with_null_values[i]] = raw_data[columns_with_null_values[i]].fillna(median_value)

raw_data
print('Number of null values in each column')

print(raw_data.isnull().sum())
def split_for_validation(a,n):

    '''

    a = dataframe,

    n = percentage of split

    '''

    return a[: len(a) - (int(len(a) * n))].copy(), a[len(a) - (int(len(a) * n)):].copy()    



# function for random forest algorithm classifier



def rand_forest_classifier(raw_data, validation_split):

    feature_columns = raw_data.iloc[:,:-1].values

    dependent_column = raw_data.iloc[:,-1].values

    X_train, X_valid = split_for_validation(a = feature_columns,

                                            n = validation_split)

    y_train, y_valid = split_for_validation(a = dependent_column,

                                            n = validation_split)

    print('Training data: ', X_train.shape, y_train.shape)

    print('Validation data: ', X_valid.shape, y_valid.shape)



    m = RandomForestClassifier(n_jobs=-1)

    m.fit(X_train, y_train)



    predTree = m.predict(X_valid)

    print()

    print('Training accuracy: ', round((m.score(X_train, y_train))*100,2),'%')

    print()

    print('Validation accuracy: ', round((m.score(X_valid, y_valid))*100,2),'%')



    from sklearn.metrics import confusion_matrix, plot_confusion_matrix

    plot_confusion_matrix(m, X_valid, y_valid)

    plt.title('Confusion Matrix')

    plt.show()

    

    
raw_data = shuffle(raw_data)

rand_forest_classifier(raw_data = raw_data,

                       validation_split = 0.2)
def log_regression(raw_data, validation_split):

    feature_columns = raw_data.iloc[:,:-1].values

    dependent_column = raw_data.iloc[:,-1].values

    X_train, X_valid = split_for_validation(a = feature_columns,

                                            n = validation_split)

    y_train, y_valid = split_for_validation(a = dependent_column,

                                            n = validation_split)

    print('Training data: ', X_train.shape, y_train.shape)

    print('Validation data: ', X_valid.shape, y_valid.shape)





    model = LogisticRegression()

    model.fit(X_train, y_train)

    predTree = model.predict(X_valid)

    print('Training accuracy: ', round((model.score(X_train, y_train))*100,2),'%')

    print()

    print('Validation accuracy: ', round((model.score(X_valid, y_valid))*100,2),'%')   

    

    from sklearn.metrics import confusion_matrix, plot_confusion_matrix

    

    plot_confusion_matrix(model, X_valid, y_valid)

    plt.title('Confusion Matrix')

    plt.show()
raw_data = shuffle(raw_data)

log_regression(raw_data = raw_data,

                       validation_split = 0.2)