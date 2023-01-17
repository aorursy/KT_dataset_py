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

import numpy as np

import matplotlib.pyplot as plt
datatrain = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

datatest = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
datatrain.head()
datatest.head()
X = datatrain.iloc[:, 2:].values

y = datatrain.target.values

X_test = datatest.iloc[:, 1:].values
X_train = X

y_train = y
from sklearn.linear_model import LogisticRegression

#create an instance and fit the model 

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_train,predictions))
sub_df = pd.DataFrame({'ID_code':datatest.ID_code.values})

sub_df['target'] = predictions

sub_df.to_csv('submission.csv', index=False)