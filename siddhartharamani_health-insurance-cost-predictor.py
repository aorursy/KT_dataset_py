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
#importing all required libraries



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.metrics import r2_score,mean_squared_error
#loading csv file for analysis



df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
#checking for null values



df.isnull().sum()
#since not all of our values are numeric, I used this library to encode values to perform visualizations and prediction



from sklearn.preprocessing import LabelEncoder



encode = LabelEncoder()



encode.fit(df.sex.drop_duplicates()) 

df.sex = encode.transform(df.sex)



encode.fit(df.smoker.drop_duplicates())

df.smoker = encode.transform(df.smoker)



encode.fit(df.region.drop_duplicates())

df.region = encode.transform(df.region)
#updated dataset with encoded values with no null values



df
#I used this to see how trends on each graph looks like  



sns.pairplot(df)
#we see here that there is some correlation between insurance charges and smoking habits

#I expected BMI and age to have more correlation with insurance charges \_(:|)_/ 



df.corr()['charges'].sort_values()
#this heatmap shows the correlation that we found in the last step. Darker squares indicated higher correlation



plt.figure(figsize = (8,6))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap = 'Blues', square = True)
#lets get started with some visualizations



plt.figure(figsize = (8,6))

sns.distplot(df['age'], bins = 10, color = 'red')
plt.figure(figsize = (8,6))

sns.countplot(x = 'sex', palette = 'husl', data = df)
plt.figure(figsize = (8,6))

sns.distplot(df['bmi'], bins = 10)
plt.figure(figsize = (8,6))

sns.countplot(x = 'children', palette = 'husl', data = df)
plt.figure(figsize=(8,6))

sns.countplot(x = 'smoker', hue = 'sex', data = df, palette = 'husl')
plt.figure(figsize=(8,6))

sns.countplot(x = 'region', data = df, palette = 'PuBu')
plt.figure(figsize = (8,6))

sns.distplot(df['charges'], bins = 10, color = 'purple')
x = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
#to perform predictions, I divided the x and y values to training and testing values to an 80/20 split



from sklearn.model_selection import train_test_split 



xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.2, random_state = 0)
#here we import the linear regression model from scikit learning 



from sklearn import linear_model

from sklearn.linear_model import LinearRegression



reg = LinearRegression()

reg.fit(xTrain, yTrain)
yPred = reg.predict(xTest)
#almost 80% accuracy, pretty good



print(reg.score(xTest,yTest))
#importing decision tree regression model 



from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor(random_state = 0)

tree.fit(xTrain, yTrain)

treeYpred = tree.predict(xTest)
#65% accuracy, we're going to pass on this one 



print(r2_score(yTest,treeYpred))
#Random forest regression for the win



from sklearn.ensemble import RandomForestRegressor



frr = RandomForestRegressor(n_estimators = 10, random_state = 0)

frr.fit(xTrain, yTrain)

frrYpred = frr.predict(xTest)
#87% accuracy, this is it!



print(r2_score(yTest,frrYpred))