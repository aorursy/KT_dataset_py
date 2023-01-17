# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
% matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print("....")

# Any results you write to the current directory are saved as output.
happiness_index = pd.read_csv("../input/2015.csv")
happiness_index.head()
happiness_index['GDP'] = happiness_index['Economy (GDP per Capita)']
happiness_index['Health'] = happiness_index['Health (Life Expectancy)']
happiness_index['Trust'] = happiness_index['Trust (Government Corruption)']
happiness_index['Dystopia'] = happiness_index['Dystopia Residual']
happiness_index.drop(['Economy (GDP per Capita)','Health (Life Expectancy)','Trust (Government Corruption)','Dystopia Residual'], axis=1, inplace=True)
happiness_index.head()

sns.pairplot(data=happiness_index,)
X = happiness_index[['Standard Error','Family','Freedom','Generosity','GDP','Health','Trust','Dystopia']]
y = happiness_index[['Happiness Score']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
print(" X_train:{} \n y_train:{} \n X_test:{} \n y_test:{}".format(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test)))
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_predicted = linear_model.predict(X_test)
plt.scatter(y_predicted, y_test)
y_ticks = np.linspace(y_predicted.min(), y_predicted.max(), 10)
print(y_tic)
plt.yticks(y_ticks)