# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv')

train_data.info()
test_data = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv')



test_data.info()
sample_sub = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv')



print(sample_sub)

print(sample_sub.head(10))
print(train_data.head(10))

print(test_data.head(10))
x = train_data.drop(columns = ['price_range'])

y = train_data['price_range']
print(x,y,sep="\n")
x_test=test_data.drop(columns=['id'])

print(x_test)
from sklearn import naive_bayes

from sklearn.model_selection import cross_val_score

nb = naive_bayes.GaussianNB()

nb = nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)

value = cross_val_score(naive_bayes.GaussianNB(), x_train, y_train, cv = 5)

print(value.mean())
from sklearn import tree

dtree = tree.DecisionTreeClassifier()

dtree = dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)

value = cross_val_score(tree.DecisionTreeClassifier(), x_train, y_train, cv = 5)

print(value.mean())
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr = lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

value = cross_val_score(LinearRegression(), x_train, y_train, cv = 5)

print(value.mean())

pred = lr.predict(test_data)

result_lr = pd.DataFrame({'id' : sample_sub['id'], 'price_range' : pred})

result_lr.to_csv('/kaggle/working/result_lr.csv', index = False)
