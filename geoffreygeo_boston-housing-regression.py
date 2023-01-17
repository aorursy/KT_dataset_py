# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#retrieving the data set 

df = pd.read_csv('/kaggle/input/bostonhoustingmlnd/housing.csv')

df.head()
sns.distplot(df['RM'])
y = df.pop('MEDV')
y.head()
x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.3,shuffle = True)

print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)
model = LinearRegression()

model.fit(x_train,y_train)
predictions = model.predict(x_test)
plt.scatter(y_test,predictions)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
print("Model Score {}".format(model.score(x_test,y_test)))
print("R2 Score {}".format(r2_score(y_test,predictions)))