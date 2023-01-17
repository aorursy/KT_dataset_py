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
# read data

test  = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

print(train.shape, test.shape)

train.head()
test.head()
y = train['SalePrice']

X = train.drop('SalePrice',axis=1)
data = X.append(test)
features = data.columns

cat_features = []

num_features = []

def feature_type (feature_list):

    for col in feature_list:

        if len(data[col].unique())<15:

            cat_features.append(col)

        else :

            num_features.append(col)

feature_type(features)

(len(cat_features), len(num_features))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
