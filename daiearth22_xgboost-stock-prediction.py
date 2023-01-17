# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# This script shows you how to make a submission using a few

# useful Python libraries.

# It gets a public leaderboard score of 0.76077.

# Maybe you can tweak it and do better...?



import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import numpy as np
# Load the data

train_df = pd.read_csv('../input/FastRetailing2016Jan-Jun-Stock-Training.csv', header=0)

test_df = pd.read_csv('../input/FastRetailing2016Jul-Dec-Stock-Test.csv', header=0)

#Below is taken from Titanic project. So below does not work as it is.



# XGBoost doesn't (yet) handle categorical features automatically, so we need to change

# them to columns of integer values.  

# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more

# details and options

le = LabelEncoder()

for feature in nonnumeric_columns:

    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])



# Prepare the inputs for the model

train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()

test_X = big_X_imputed[train_df.shape[0]::].as_matrix()

train_y = train_df['Survived']



# You can experiment with many other options here, using the same .fit() and .predict()

# methods; see http://scikit-learn.org

# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(test_X)



# Kaggle needs the submission to have a certain format;

# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv

# for an example of what it's supposed to look like.

submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)


