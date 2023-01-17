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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.sex.replace('female', 0, inplace = True)
df.sex.replace('male', 1, inplace = True)
df.smoker.replace('no', 0, inplace = True)
df.smoker.replace('yes', 1, inplace = True)
df.region.value_counts()
df.head()
df.isna().sum()
df.region.replace('northeast', 0, inplace = True)
df.region.replace('northwest', 1, inplace = True)
df.region.replace('southeast', 2, inplace = True)
df.region.replace('southwest', 3, inplace = True)
sns.catplot(x = 'sex', y = 'charges', data = df, kind = 'box')
# The men have a larger range of charges even though the median for both sexes is around the same. 
#I confirmed in the cell below that it wasn't because more men went to the hospital
# it could be that the range of issues they're dealing with incurred a more varied amount of charges than women.
df.sex.value_counts()
sns.catplot(x = 'smoker', y = 'charges', data = df, kind = 'box')
sns.scatterplot(df.bmi, df.charges, data = df)
# No clear trend, but there are 2 clusters that emerge: the bigger lower bottom and the upper one
sns.scatterplot(df.bmi, df.charges, data = df, hue = 'smoker')
# There is a definite trend where the smokers are highlighted.
sns.catplot('children', 'charges', data = df, kind = 'box')
sns.scatterplot(df.age, df.charges, data = df)
# Generally, the charges are rising with age
sns.scatterplot(df.age, df.charges, data = df, hue = 'smoker')
sns.scatterplot(df.age, df.charges, data = df, hue ='bmi')
#Generally, the higher bmi's have higher charges within the same age range
df.head()
sns.scatterplot(df.age, df.charges, data = df, hue ='sex')
# I can't tell too much of a  pattern here, where there are children involved.
sns.scatterplot(df.age, df.charges, data = df, hue ='children')
# I can't tell too much of a  pattern here, where there are children involved.
sns.scatterplot(df.age, df.charges, data = df, hue ='region')
#There's no pattern where it comes to region, either
sns.catplot('region', 'charges', data = df, kind = 'box')
# Region doesn't seem to be playing a role
df.corr()
sns.heatmap(df.corr())
#There's a stong correlation between smokers and the charges
# Age with the charges , and then bmi with the charges, as well

# Because children, sex and region do not appear to have a correlation, I'll drop them.
df.drop(['sex','region'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
x = df.iloc[:, :-1]
x.head()
y = df.iloc[:, -1:]
y.head()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 1, test_size = 0.2)
lr = LinearRegression()
dt = DecisionTreeClassifier(random_state = 1)
gr = GradientBoostingRegressor(learning_rate = 0.1)
pr = PolynomialFeatures(degree = 2)
lr.fit(xtrain, ytrain)
lr.score(xtrain, ytrain)
lr.score(xtest, ytest)
gr.fit(xtrain, ytrain)
gr.score(xtrain, ytrain)
gr.score(xtest, ytest)
pr = PolynomialFeatures (degree = 2)
x_quad = pr.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x_quad,y, random_state = 0, test_size = 0.2)

plr = LinearRegression().fit(xtrain,ytrain)

Y_train_pred = plr.predict(xtrain)
Y_test_pred = plr.predict(xtest)

print(plr.score(xtest, ytest))
estimators = [('gr', gr),('plr', plr)]
sr = StackingRegressor(estimators=estimators)
sr.fit(xtrain, ytrain)
sr.score(xtrain, ytrain)
sr.score(xtest, ytest)
estimators = [('plr', plr),('gr', gr)]
sr = StackingRegressor(estimators=estimators)
sr.fit(xtrain, ytrain)
sr.score(xtrain, ytrain)
sr.score(xtest, ytest)
