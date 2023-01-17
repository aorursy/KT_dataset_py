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
import seaborn as sns

import scipy.stats as stat
data = pd.read_csv("../input/train.csv", na_values=['NaN'])



data = data.dropna()



data.tail()
data.columns
data['Sex'] = data['Sex'].map({'male':1, 'female':2}).astype(int)

data['Embarked'] = data['Embarked'].map({'S':1, 'C':2, 'Q':3}).astype(int)

data['Survived'] = data['Survived'].map({0:1, 1:2})

data.tail()
stat.chisquare(data['Pclass'], data['Embarked'])