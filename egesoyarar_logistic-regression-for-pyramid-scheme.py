# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/pyramid-scheme-profit-or-loss/pyramid_scheme.csv")
data.head()
data.drop(["Unnamed: 0","cost_price","sales_commission"] , axis = 1, inplace = True) #axis = 1 for column, axis = 0 for row deleting !!!
data.head()

data.profit = [1 if money>0 else 0 for money in data.profit]

sns.countplot(x="profit", data=data) #to visualize #of profits and losses in barchart by usign seaborn lib.
data.loc[:,'profit'].value_counts()
y = data.profit.values
x_data = data.drop(["profit"],axis = 1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.head()
from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
x_train.head()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train.T , y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))