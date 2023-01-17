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
import os 

import keras 

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 
df=pd.read_csv('/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv')

df.head()
df.rename(columns = {'total_sqft':'m2'},inplace=True)

df['m2'].dtypes
df['m2'] = pd.to_numeric(df['m2'], errors='coerce') 
df.dtypes
print(df.isnull().sum())
df["m2"].fillna(df["m2"].mean(), inplace=True)
print(df.isnull().sum())
import seaborn as sns

sns.pairplot(df)
sns.heatmap(df.corr()) 
x = df[['m2']]

y = df['price']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train,y_train)
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])

coeff_df
lm.score(x_test, y_test)
#Grafik eksenlerini belirledik

df.plot(kind='scatter',x='m2',y='price', title='Housing Prices and Square Footage of Bengaluru')



#Fit kare bazında tahmini konut fiyatını bulduk.

y_pred = lm.predict(x) 



#Linear regression çizgisini çizdik.

plt.plot(x, y_pred, color='red')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
new_prediction = 5

new_prediction = new_prediction.reshape(1,-1)

print(lm.predict([new_prediction].reshape))