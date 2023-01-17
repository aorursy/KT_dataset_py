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
# Read the data

data_df=pd.read_csv("/kaggle/input/insurance/insurance.csv")



# See sample data

data_df.head()
# Check for the dtypes & nulls

data_df.info()
# Just being doubly sure

data_df.isnull().sum()
# Getting some stats about the data

data_df.describe()
# Getting the numerical columns first

data_df.describe().columns
# Separating numerical and categorical data

df_num = data_df[['age','bmi','children','charges']]

df_cat = data_df[['sex','smoker','region']]
# Converting the categorical data into numeric data.

# One hot enconding the categorical data

df1 = pd.get_dummies(df_cat)

df1
# Concatenating the categorical and numerical data to form our data set.

data = pd.concat([df_num,df1], axis=1)

data
# Finding out the correlation between the features

data.corr()
# Heatmap to show correlation

sns.heatmap(data.corr(), cmap='RdBu')
# Correlation between charges and the other features.

data.corr()['charges'].sort_values()
# Graph showing the min and maximum charges

count, bin_edges = np.histogram(data['charges'])

data['charges'].plot(kind='hist', xticks=bin_edges, figsize=(20,12))

plt.title("Patient Charges")

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
x = data.drop(['charges'], axis = 1)

y = data['charges']



x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state = 0)

lr = LinearRegression().fit(x_train,y_train)
y_train_pred = lr.predict(x_train)

y_test_pred = lr.predict(x_test)



print(lr.score(x_test,y_test))
from sklearn.metrics import r2_score,mean_squared_error
print('MSE train data:' , mean_squared_error(y_train,y_train_pred))

print('MSE test data:' , mean_squared_error(y_test,y_test_pred))



print('R2 train data:', r2_score(y_train,y_train_pred))

print('R2 test data:' , r2_score(y_test,y_test_pred))