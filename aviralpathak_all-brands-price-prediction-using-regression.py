# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files = ['audi','bmw','ford','hyundi','merc','skoda','toyota','vauxhall','vw']



frames = []



for f in files:

    

    frame = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/'+f+'.csv')

    

    frames.append(frame)
df = pd.concat(frames, sort=False)
df.reset_index(drop=True, inplace=True)
df
df.info()
#tax(E) column is a tax column for dataframe hyundai and now we want to merge both of them. they both have NaN rows and they have the values for each others NaN values, so when their is a NaN value in the tax or tax(E) column the NaN will get remove and the value will come.



df['tax'] = df["tax"].fillna('').map(str) + df["tax(£)"].fillna('').map(str)
df.drop(['tax(£)'], axis = 1, inplace = True)
df['tax'] = df['tax'].astype('float64')
#wrong data



df[df['year'] == 2060]

df = df.drop(39175)
sns.pairplot(df)
plt.figure(figsize=(12,8))

plt.title('year')

plt.ylabel('# of games position picked')

sns.countplot(df['year'])
sns.catplot(x = 'year', y= 'price', data = df, kind='point', aspect=4);
plt.figure(figsize=(15,5),facecolor='w') 

sns.scatterplot(df["mileage"], df["price"], hue = df["fuelType"])
plt.figure(figsize=(15,5),facecolor='w') 

sns.boxplot(x = 'price', y ='transmission',data=df, palette="Set3")
df_cat = df[['transmission','fuelType']]
df_cat = pd.get_dummies(df_cat)

df_cat
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_num = df[['mileage','tax','mpg','engineSize']]
df_num[['mileage','tax','mpg','engineSize']] = scaler.fit_transform(df_num[['mileage','tax','mpg','engineSize']])
df_num
df1 = pd.concat([df_num,df_cat,df['price']],axis=1)
df1.head()
from sklearn.model_selection import train_test_split
X = df1.drop("price", axis = 1)

y = df1["price"]
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)

y_pred =lin_reg.predict(X_test)

# Calculating RMSE

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

r2score = r2_score(y_test,y_pred)
print("R2 score is ", r2score)

print("rmse is ", rmse)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train,y_train)
forest_y_pred = forest.predict(X_test)

# Calculating RMSE

forest_rmse = np.sqrt(mean_squared_error(y_test,forest_y_pred))

forest_r2score = r2_score(y_test,forest_y_pred)

print("R2 score is ", forest_r2score)

print("rmse is ", forest_rmse )