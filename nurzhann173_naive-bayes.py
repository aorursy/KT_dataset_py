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
x = datatrain.iloc[:, 2:].values

y = datatrain.target.values

x_test = datatest.iloc[:, 1:].values
x_train = x

y_train = y
x_test
x_test.shape

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
gauss = GaussianNB()

y_prediction = gauss.fit(x_train, y_train).predict(x_test)
print(classification_report(y_train,y_prediction))
sub_df = pd.DataFrame({'ID_code':datatest.ID_code.values})

sub_df['target'] = y_prediction

sub_df.to_csv('submission.csv', index=False)