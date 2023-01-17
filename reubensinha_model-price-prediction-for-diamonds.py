# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/diamonds.csv')
cat_features = ['cut', 'color', 'clarity']
df[cat_features] = df[cat_features].apply(lambda x: x.astype('category'))

clarity_order = ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF')
cut_order = ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal')
color_order = ('J','I','H','G','F','E','D')

df['clarity'] = pd.Categorical(df.clarity, ordered = True, categories = clarity_order)
df['cut'] = pd.Categorical(df.cut, ordered = True, categories = cut_order)
df['color'] = pd.Categorical(df.color, ordered = True, categories = color_order)

dummy_df = pd.get_dummies(df)
X = dummy_df.drop(['price'], axis = 1)
y = dummy_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
grid_parameters = {'alpha': [0.1, 1, 1.5, 5, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9] }
model = GridSearchCV(ElasticNet(), grid_parameters, cv = 3)
model.fit(X_train, y_train)
# Any results you write to the current directory are saved as output.
model.score(X_test, y_test)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.style.use('ggplot')
import os
print(os.listdir("../input"))

df = pd.read_csv('../input/diamonds.csv')
cat_features = ['cut', 'color', 'clarity']
df[cat_features] = df[cat_features].apply(lambda x: x.astype('category'))

lr = LinearRegression()
lr.fit(df.carat.values.reshape(-1,1), df.price.values.reshape(-1,1))

fig = plt.figure()
axis = fig.add_subplot(1,1,1)
plt.scatter(df.carat, df.price, alpha = 0.4, s = 3)
axis.set_xlabel('Carat')
axis.set_ylabel('Price')
axis.set_title('Carat vs Price')
plt.plot(np.arange(0, 4), lr.predict(np.arange(0,4).reshape(-1,1)), color = 'b')
plt.savefig("Carat-vs-Price.jpg")
