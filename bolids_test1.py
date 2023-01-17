# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv', header = 0)
train.info()
test = pd.read_csv('../input/test.csv', header = 0)
test.info()

def data_cleaning (df):
    # create a new column as "Gender" and replace 'female' and 'male' as 0 and
    # 1, respectively
    df['Gender'] = df['Sex'].map({'female' : 0, 'male' : 1}).astype(int)

    # do the same to "Embarked"
    # df['Boarding'] = df['Embarked'].map({'S' : 0, 'C' : 1, 'Q' :
    # 2}).astype(int)

    # df.apply(lambda x : sum(x.isnull()), axis = 0)

    # use the pivot table to calculate the median of 'Age' based on 'Gender'
    # and 'Pclass', learned this trick from
    # http://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
    pv_table = df.pivot_table (values='Age', index='Gender', \
            columns='Pclass', aggfunc=np.median)

    # make a copy of 'Age'
    df['AgeFill'] = df['Age']

    # OMG I hate this freaking long line
    df['AgeFill'].fillna(df[df['AgeFill'].isnull()].apply(lambda x :
        pv_table.loc[x['Gender'], x['Pclass']], axis=1), inplace=True)

    # create a feature of recording that the 'Age' was originally missing
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    # Combining 'SibSp' and 'Parch'
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Artificial feature of 'Age * Pclass'
    df['Age*Class'] = df.AgeFill * df.Pclass

    # This shows which ones are 'objects' that we will drop
    df.dtypes[df.dtypes.map(lambda x: x=='object')]
    df = df.drop(['Name', 'Sex','Ticket', 'Cabin', 'Embarked', 'Age'], \
            axis=1)
    return df

train = data_cleaning(train).drop(['PassengerId'], axis=1)
train_data = train.values
# train.info()

test_copy = data_cleaning(test.copy()).drop(['PassengerId'], axis=1)
test_copy['Fare'].fillna(test_copy['Fare'].median(), inplace=True)
test_data = test_copy.values
# test.info()


# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100)

forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

output = forest.predict(test_data)
print(train_data[0::, 1::])
print (forest.score(train_data[0::, 1::], train_data[0::,0]))

# make a data frame for saving as csv file
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(output, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])
# Any results you write to the current directory are saved as output.

 
