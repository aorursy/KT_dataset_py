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
df = pd.read_csv("/kaggle/input/bmw-car-prices/bmw_carprices.csv")
df.head()
#shape of the dataset

df.shape
df.describe()
df.info()
#finding correlation between variables

df.corr()
#plotting heat map for the correlation



plt.figure(figsize=(8, 8))

sns.heatmap(df.corr(), annot=True)

plt.show()
#plot between mileage and selling price



plt.scatter(df['Mileage(kms)'], df['Sell Price($)'])

plt.xlabel('Mileage(kms)')

plt.ylabel('Sell Price($)')

plt.show()
#plot between age and selling price



plt.scatter(df['Age(yrs)'], df['Sell Price($)'])

plt.xlabel('Age(yrs)')

plt.ylabel('Sell Price($)')

plt.show()
#plot between mileage and age in response to selling price



plt.scatter('Mileage(kms)', 'Sell Price($)', c = 'Age(yrs)' ,data = df)

plt.xlabel('Mileage(kms)')

plt.ylabel('Sell Price($)')

plt.colorbar().set_label( 'Age' )

plt.colormaps()

plt.show()
#scalarize the data 



from sklearn.preprocessing import StandardScaler



col = ['Mileage(kms)', 'Age(yrs)', 'Sell Price($)']



scaler = StandardScaler()



df[col] = scaler.fit_transform(df[col])



#df.head()
#Splitting dataset and X and Y set

#Not splitting to test-train as the data is less in quantity

feature_col = ['Mileage(kms)', 'Age(yrs)']

x = df[feature_col]

y = df['Sell Price($)']
#fitting a linear regression model



from sklearn.linear_model import LinearRegression



model= LinearRegression()

model.fit(x, y) 
# Predict the value



y_pred = model.predict(x)
#score/accuracy of the model



model.score(x, y)
# printing values

print('Slope:' ,model.coef_)

print('Intercept:', model.intercept_)

list(zip(col, model.coef_))
from sklearn.metrics import mean_squared_error



rmse = mean_squared_error(y, y_pred)



print('Root mean squared error: ', np.sqrt(rmse))