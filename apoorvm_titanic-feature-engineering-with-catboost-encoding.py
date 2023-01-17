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
from sklearn.model_selection import train_test_split

import random

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
import itertools

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
import itertools
def Feature_selection(X):

    cat_features = [y for y in X.columns if X[y].dtypes == 'object']

#     print(cat_features)

    cat_features = [col for col in cat_features if col not in ['Name', 'Ticket', 'Cabin']]

    

    interactions = pd.DataFrame(index=X.index)

#     print(interactions)

    

    for col1, col2 in itertools.combinations(cat_features, 2):

        new_col_name = '_'.join([col1, col2])

        new_values = X[col1].map(str) + "_" + X[col2].map(str)

#     print(new_values)



        encoder = LabelEncoder()

        X[new_col_name] = encoder.fit_transform(new_values)

        

        

#     print(interactions.head())

#     print(X)

    

#     feature_cols = X.columns.drop('Survived')

    

#     selector = SelectKBest(f_classif, k=5)

    

#     X_new = selector.fit_transform(X[feature_cols], X['Survived'])

# #     print(X_new)

#     selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

#                                  index=X.index, 

#                                  columns=feature_cols)

# #     print(selected_features.head())

#     # Dropped columns have values of all 0s, so var is 0, drop them

#     selected_columns = selected_features.columns[selected_features.var() != 0]



#     # Get the valid dataset with the selected features.

# #     print(X[selected_columns].head())

    return X
import category_encoders as ce
# Fit the encoder using the categorical features and target

#     encoder.fit(X[cat_features], X['Survived'])

    



    # Apply the label encoder to each column

    #     df2 = df2.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')



#     print(encoded.head(10))

def Categorical_Encoding_train(X):

#     print(X.head())

    cat_features = [y for y in X.columns if X[y].dtypes == 'object']

#     print(cat_features)

    cat_features = [col for col in cat_features if col not in ['Name']]

    

    encoder = ce.CatBoostEncoder(cols = cat_features)

    

    

    

    encoder.fit(X[cat_features], X['Survived'])

    



    

    numerical_col = [y for y in X.columns if X[y].dtypes != 'object']

#     print(encoded.shape)

    

    data = X[numerical_col].join(encoder.transform(X[cat_features]))

    return data, encoder

    
def Categorical_Encoding_test(X, encoder):

    cat_features = [y for y in X.columns if X[y].dtypes == 'object']

    print(cat_features)

    cat_features = [col for col in cat_features if col not in ['Name']]

    

    numerical_col = [y for y in X.columns if X[y].dtypes != 'object']

#     print(encoded.shape)

    

    data = X[numerical_col].join(encoder.transform(X[cat_features]))

    print(data.head())

    return data
from sklearn.experimental import enable_iterative_imputer  

from sklearn.impute import IterativeImputer
def iterative_imputer(X):

    numerical_feature = [y for y in X.columns if X[y].dtypes != 'object' 

                         and X[y].isnull().sum() != 0]

    

    imp_mean = IterativeImputer(max_iter=10, verbose=0)

    X[numerical_feature] = imp_mean.fit_transform(X[numerical_feature])

    

    return X
def numerical_imputer(X):

    

    numerical_feature = [y for y in X.columns if X[y].dtypes != 'object' 

                         and X[y].isnull().sum() != 0]

#     print(numerical_feature)

    my_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    X[numerical_feature] = pd.DataFrame(my_imputer.fit_transform(X[numerical_feature]))

    return X
import lightgbm as lgb
def train_and_predict(train_X, train_y, valid_X, valid_y, X_test):

#     feature_cols = train.columns.drop('outcome')



    dtrain = lgb.Dataset(train_X, label=train_y)

    dvalid = lgb.Dataset(valid_X, label=valid_y)



    param = {'num_leaves': 64, 'objective': 'binary'}

    param['metric'] = 'auc'

    num_round = 1000

    bst = lgb.train(param, dtrain, num_round,

                    valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

    

    #get predictions

    y_pred = bst.predict(X_test)

    

    return y_pred
if __name__ == '__main__':

    seed = 123

    random.seed(seed)

    

    print ('Loading Training Data')

    baseline_data = pd.read_csv('/kaggle/input/titanic/train.csv')

    

   

    baseline_data = Feature_selection(baseline_data)

    

    baseline_data = numerical_imputer(baseline_data)

    

#     print(baseline_data[numerical_feature].head(10))

    

    

    encoded_data, encoder = Categorical_Encoding_train(baseline_data)

#     print(encoded_data.head(10))



    

    cols = [col for col in encoded_data.columns if col not in ['Survived','PassengerId']]

#     print("Cols", cols)

    X = encoded_data[cols]

    y = baseline_data['Survived']

    

#     print(baseline_data.isnull().sum())

    valid_fraction = 0.1

    valid_size = int(len(encoded_data) * valid_fraction)



    train_X = X[:-2 * valid_size]

    train_y = y[:-2 * valid_size]

    valid_X = X[-2 * valid_size:]

    valid_y = y[-2 * valid_size:]

    

    #Now Laoding the testing data

    print ('Loading Testing Data')

    test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

    

    test_data = Feature_selection(test_data)

    

    test_data = numerical_imputer(test_data)

    

    encoded_data_test = Categorical_Encoding_test(test_data, encoder)

#     print(encoded_data_test[cols])

# #     print(encoded_data_test[cols].head())

    

    X_test = encoded_data_test[cols].iloc[:,:]

    print(X_test.head())

    

    y_pred = train_and_predict(train_X, train_y, valid_X, valid_y, X_test)

    

    y_pred = np.around(y_pred)

    y_pred = y_pred.astype(int)

    # Save test predictions to file

    output = pd.DataFrame({'PassengerId': test_data.PassengerId,

                           'Survived': y_pred})

    output.to_csv('submission_grid_search.csv', index=False)

    

    

# #     Feature_selection(baseline_data)
# [x for x in baseline_data.columns if baseline_data[x].dtypes == 'object']

# [x in baseline_data.columns if x.=='object']

# baseline_data.isnull().sum()

# baseline_data['Cabin'].head(10)

# baseline_data.groupby('Survived')['Pclass'].count()

# baseline_data.groupby('Age')['Survived'].count()

# # encoder = LabelEncoder()



#     # Apply the label encoder to each column

# encoded = pd.DataFrame([1.0, 'a']).apply(encoder.fit_transform)

# # encoded = encoder.fit_transform(baseline_data['Cabin'][:2])
baseline_data.groupby('Pclass')['Survived'].count()
baseline_data.groupby('Embarked')['Survived'].count()
baseline_data.Age