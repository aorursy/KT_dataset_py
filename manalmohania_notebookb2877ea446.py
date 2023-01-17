# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



# Any results you write to the current directory are saved as output.
# read csv and remove pId, name, ticket, fare, cabin, embarked

# normalise as follows

# sex -> 0 to 2

# age -> 0 to 2

# sib sp, parch -> 0 to 1



# noramalise columns of a df in the range [0, upper_val]

def normalise_col(df, col_names, upper_vals):

    assert len(col_names) == len(upper_vals)

    res = df.copy()

    for i in range(0, len(col_names)):

        col_name = col_names[i]

        upper_val = upper_vals[i]

        col_max = res[col_name].max()

        col_min = res[col_name].min()

        res[col_name] = upper_val * (df[col_name] - col_min)/(col_max - col_min)

    return res



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

survived = df_train['Survived']

df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_train.replace(to_replace='male', value=0, inplace=True)

df_test.replace(to_replace='male', value=0, inplace=True)

df_train.replace(to_replace='female', value=2, inplace=True)

df_test.replace(to_replace='female', value=2, inplace=True)

df_train.fillna(df_train.mean(), inplace=True)

df_test.fillna(df_test.mean(), inplace=True)

df_train = normalise_col(df_train, ['Age', 'SibSp', 'Parch'], [2, 1, 1])

df_test = normalise_col(df_test, ['Age', 'SibSp', 'Parch'], [2, 1, 1])
# performing knn

knn = KNeighborsRegressor(n_neighbors=29)

knn.fit(df_train, survived)

predictions = knn.predict(df_test)

for i in range(len(predictions)):

    if predictions[i] >= 0.5:

        predictions[i] = 1

    else:

        predictions[i] = 0

predictions = predictions.astype(int)
# write to csv

pId = []

for i in range(len(predictions)):

    pId.append(892 + i)

data = {'PassengerId' : pId, 'Survived': predictions}

final = pd.DataFrame(data=data)

final.to_csv('submission.csv', index=False)