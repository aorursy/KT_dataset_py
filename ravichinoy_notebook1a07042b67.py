# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np

import pandas as pd



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Any results you write to the current directory are saved as output.



train_data = harmonize_data(train)

test_data  = harmonize_data(test)

from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation

from sklearn import tree

def harmonize_data(titanic):

    

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic["Age"].median()

    

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    

    titanic["Embarked"] = titanic["Embarked"].fillna("S")



    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())



    return titanic

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



alg = RandomForestClassifier(

    random_state=1,

    n_estimators=500,

    min_samples_split=4,

    min_samples_leaf=2

)



scores = cross_validation.cross_val_score(

    alg,

    train_data[predictors],

    train_data["Survived"],

    cv=3

)



print(scores.mean())

i_tree = 0

for tree_in_forest in alg.estimators_:

    with open('../input/tree_' + str(i_tree) + '.dot', 'w') as my_file:

        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)

    i_tree = i_tree + 1
