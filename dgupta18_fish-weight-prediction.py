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
df = pd.read_csv(os.path.join(dirname, filename))

df.head(10)
df.rename(columns={'Length1':'Vertical Length','Length2':'Diagonal Length','Length3':'Cross Length'},inplace=True)
df
def species_int(col):

    if col == 'Bream':

        return 1

    elif col == 'Perch':

        return 2

    elif col == 'Roach':

        return 3

    elif col == 'Pike':

        return 4

    elif col == 'Smelt':

        return 5

    elif col == 'Parkki':

        return 6

    elif col == 'Whitefish':

        return 7
df['SpeciesNo'] = df['Species'].apply(species_int)
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns 
sns.countplot(x = df['Species'])
sns.scatterplot(x = 'Weight',y='Vertical Length',data = df)
sns.scatterplot(x = 'Weight',y='Diagonal Length',data = df)
sns.scatterplot(x = 'Weight',y='Cross Length',data = df,hue = 'Species',palette='coolwarm')
sns.pairplot(df)
from sklearn.model_selection import train_test_split

X = df.drop(labels=['SpeciesNo','Weight','Species'],axis=1)

y = df['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

prediction = lm.predict(X_test)
lm.score(X_test,y_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

print('MAE:',mean_absolute_error(prediction,y_test))

print('MSE:',mean_squared_error(prediction,y_test))

print('R2 Score:',r2_score(prediction,y_test))
plt.scatter(prediction,y_test)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
poly.fit(X_train,y_train)
xtrain_poly = poly.transform(X_train)

xtest_poly = poly.transform(X_test)
lm1 = LinearRegression()
lm1.fit(xtrain_poly,y_train)
poly_predict = lm1.predict(xtest_poly)
lm1.score(xtest_poly,y_test)
print('MAE:',mean_absolute_error(poly_predict,y_test))

print('MSE:',mean_squared_error(poly_predict,y_test))

print('R2 Score:',r2_score(poly_predict,y_test))
fig = plt.figure(figsize=(12,8))

plt.scatter(poly_predict,y_test,edgecolors='black')