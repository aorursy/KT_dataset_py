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
df2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

#Presense of missing values
missing = df2015.isnull().sum().sort_values(ascending=False)
print(missing)

import seaborn as sns
cor_matrix = df2015.corr()
k = 10 # the number off variables in the matrix
cols = cor_matrix.nlargest(k, 'Happiness Score')['Happiness Score'].index
cm = np.corrcoef(df2015[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

features = ['Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom', 'Dystopia Residual']
x = df2015[features]
y = df2015['Happiness Score']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1)
print(x_train.shape)
print(x_test.shape)
from sklearn.linear_model import HuberRegressor
clf = HuberRegressor()
clf.fit(x_train, y_train) # training the model

clf.predict(x_test)
my_score = clf.score(x_test, y_test)
print(my_score)
df2016 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')
sns.kdeplot(df2015['Economy (GDP per Capita)'], color='red')
sns.kdeplot(df2016['Economy (GDP per Capita)'], color= 'blue')

