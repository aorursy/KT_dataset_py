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

data = pd.read_csv("/kaggle/input/apndcts/apndcts.csv")

data.head()





from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

data_y = data['class']

data_y.head()
data_X = data.drop(['class'], axis=1)

data_X.head()
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3)

classifier = DecisionTreeClassifier("gini")

classifier.fit(train_X, train_y)

classifier.score(test_X, test_y)