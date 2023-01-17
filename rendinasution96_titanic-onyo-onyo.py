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
cover = pd.read_csv("../input/train.csv")

cover.info()
cover['Cabin'].fillna('B96B98',inplace=True)

cover.isnull().sum()
cover['Age'].fillna(cover.Age.mean(),inplace=True)

cover.isnull().sum()
cover['Cabin']
cover.isnull().sum()
cover['Embarked'].describe()
cover['Embarked'].fillna('S',inplace=True)

cover.isnull().sum()
cover
cover["Survive"]=cover["Survived"]

del cover["Survived"]
cover
X = cover[cover.columns[0:11]]

Y = cover["Survive"]

olist = list(X.cover(["object"]))

for col in olist:

    X[col]= X[col].astype("category").cat.codes