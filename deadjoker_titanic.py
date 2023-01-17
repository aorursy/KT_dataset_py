# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



# Any results you write to the current directory are saved as output.
train_df.head()
train_df[['Sex', 'Age']]
target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values



my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one,target)



print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one, target))