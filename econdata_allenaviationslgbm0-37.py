# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

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



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df

import matplotlib.pyplot as plt

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=True,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
train = import_data("../input/train.csv")

test = import_data("../input/test.csv")
test_id = test['id']

test.drop(['id'], axis=1, inplace=True)
import lightgbm as lgb

dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

dic1 = {'CA':0,'DA':1,'SS':3,'LOFT':4}

train["event"] = train["event"].apply(lambda x: dic[x])

train["event"] = train["event"].astype('int8')

train['experiment'] = train['experiment'].apply(lambda x: dic1[x])

test['experiment'] = test['experiment'].apply(lambda x: dic1[x])



train['experiment'] = train['experiment'].astype('int8')

test['experiment'] = test['experiment'].astype('int8')



y = train['event']

train.drop(['event'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.4, random_state=42)
from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE().fit_resample(X_train, y_train.ravel())
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import VotingClassifier

import lightgbm as lgb





clf1 = lgb.LGBMClassifier(

        n_estimators=100,

        learning_rate=0.04)

clf2 = lgb.LGBMClassifier(

        n_estimators=300,

        learning_rate=0.01)

clf3 = lgb.LGBMClassifier(

        n_estimators=800,

        learning_rate=0.03)
clf1.fit(X_train, y_train)

pred1 = clf1.predict(X_test)

print("lgbm1: ", accuracy_score(pred1, y_test))



pred = clf1.predict_proba(test)

sub = pd.DataFrame(pred,columns=['A', 'B', 'C', 'D'])

sub['id'] = test_id

cols = sub.columns.tolist()

cols = cols[-1:] + cols[:-1]

sub = sub[cols]

sub.to_csv("sub_lgb1.csv", index=False)