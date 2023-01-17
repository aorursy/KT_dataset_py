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

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import ADASYN





train_df = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")

test_df = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")



def clean_data(df):

    df['v2a1'] = df['v2a1'].fillna(0)

    df['v18q1'] = df['v2a1'].fillna(0)

    df['rez_esc'] = df['v2a1'].fillna(4)

    df['meaneduc'] = df['meaneduc'].fillna(test_df['meaneduc'].mean())

    df['SQBmeaned'] = df['SQBmeaned'].fillna(test_df['SQBmeaned'].mean())

    

    try:

        X = df.drop(columns = ['Target', 'Id',

                               'idhogar','dependency',

                               'edjefe','edjefa'])

        y = df['Target']

        return X, y

    

    except:

        X = df.drop(columns = ['Id',

                               'idhogar','dependency',

                               'edjefe','edjefa'])

        return X   





ada = ADASYN(random_state = 199)

X_train, y_train = clean_data(train_df)

X_train, y_train = ada.fit_sample(X_train, y_train)



X_test = clean_data(test_df)



clf = RandomForestClassifier(n_estimators=30, max_depth=10,

                             random_state=0)

clf.fit(X_train, y_train)

y_preds_rf =clf.predict(X_test)



test_df['Target']=y_preds_rf

submission = test_df[['Id', 'Target']]
submission.to_csv('submission.csv', index=False)
