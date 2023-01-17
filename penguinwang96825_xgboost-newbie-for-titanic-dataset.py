import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df



def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates = True, keep_date_col = True)

    df = reduce_mem_usage(df)

    return df
print('-' * 80)

print('train')

train = import_data('../input/train.csv')



print('-' * 80)

print('test')

test = import_data('../input/test.csv')
train.head(10)
test.head(10)
def preprocess_dataframe(df):

    df['Fare'] = df['Fare'].fillna(df['Fare'].mean()).astype("float")

    df['Age'] = df['Age'].fillna(df['Age'].mean()).astype("int")

    df['Embarked'] = df['Embarked'].cat.add_categories("Unknown").fillna("Unknown")

    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0).astype("int")

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'Unknown': 3} ).astype("int")

    df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis = 1)

    return df
cleaned_train = preprocess_dataframe(train)

cleaned_test = preprocess_dataframe(test)
cleaned_train.head(10)
def xgboost_training(df):

    train_x = df.drop('Survived', axis = 1)

    train_y = df.Survived

    (train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, 

                                                          test_size = 0.8, 

                                                          random_state = 17)

    dtrain = xgb.DMatrix(train_x, label = train_y)

    param = {'max_depth': 3, 

             'eta': 0.08, 

             'learning_rate': 0.045, 

             'gamma': 1,

             'silent': True, 

             'objective': 'binary:logistic'}

    num_round = 2

    bst = xgb.train(param, dtrain, num_round)

    preds = bst.predict(xgb.DMatrix(test_x))

    print(accuracy_score(preds.round(), test_y))



    return bst



def xgboost_predict(bst, df):

    return bst.predict(xgb.DMatrix(df))
bst = xgboost_training(cleaned_train)
from xgboost import plot_tree

import matplotlib.pyplot as plt



xgb.plot_tree(bst, num_trees = 0, rankdir = "TD")

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

plt.show()
y_pred = xgboost_predict(bst, cleaned_test).round().astype("int")
submit_data = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})

submit_data.to_csv('./submission.csv', header = True, index = None)