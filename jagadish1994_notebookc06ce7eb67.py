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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
##summary

train.describe()
train.head(10)
test.head(10)
train.shape
test.shape
print ("Skw:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
plt.matshow(train.corr())
import seaborn as sns

corr = train.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.barplot(train.OverallQual,train.SalePrice)
missing = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

missing.columns = ['Null Count']

missing.index.name = 'Feature'

missing
train.SalePrice = np.log1p(train.SalePrice )

y = train.SalePrice
train.shape 

model = lr.fit(X_train, y_train)