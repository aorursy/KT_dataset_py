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
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("/kaggle/input/boston-house-prices/housing.csv")
data.head()
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df_data = pd.read_csv("/kaggle/input/boston-house-prices/housing.csv",header = None,delimiter=r"\s+",  names = column_names)
df_data.head()
df_data.keys()
#Calling the .shape function to get the shape of the dataset
df_data.shape
# calling the describe function
df_data.describe()
df_data.isnull().sum()
df_data.isna().sum()
df_data['target']=df_data.MEDV
df_data.head()
# minimum price of tha data
minimum_pricce = df_data['target'].min()
# maximum price of data
maximum_price = df_data['target'].max()
# mean price of the data
mean_price = df_data['target'].mean()
# median price of the data
median_price = df_data['target'].median()
# standard deviation price of the data
std_price = df_data['target'].std()
#first_quartile = np.quantile(target, 25)
#third_quartile = np.quantile(target, 75)
#inter_quartile = third_quartile -  first_quartile

print("statistics for boston housing dataset: \n")
print("Minimum price: $",minimum_pricce)
print("Maximum price: $",maximum_price)
print("Mean price: $",mean_price)
print("Median price $",median_price)
print("Standard deviation of prices: $",std_price)
#print("First quartile of prices: $",first_quartile)
#print("Second quartile of prices: $",third_quartile)
#print("Interquartile (IQR) of prices: $",inter_quartile)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df_data['target'], bins = 30)
plt.show()
print(df_data[df_data['target'] == 50].shape)
df_data[df_data['target'] == 50]
df_data = df_data[df_data['target'] != 50]
prices = df_data['target']
df_data.shape
sns.set(style='ticks', color_codes=True)
plt.figure(figsize=(24, 14))
sns.heatmap(df_data.astype(float).corr(), linewidths=0.1, square=True, linecolor='white', annot=True)
plt.show()
filter_data=pd.DataFrame(df_data[['RM','LSTAT','PTRATIO','target']])
filter_data.head()
features=filter_data.drop('target',axis=1)
features.head()
features.columns
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))

for i, col in enumerate(features.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, 3, i+1)
    x = filter_data[col]
    y = prices
    plt.plot(x, y, 'o',color='pink')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),color='blue')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in features.items():
    sns.boxplot(y=k, data=features, ax=axs[index])
    index += 1
    plt.title(k)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(filter_data))
print(z)
z.max()
print(np.where(z>3))
filtered_data = filter_data[(z < 3).all(axis=1)]
print(filtered_data.shape)
print(filter_data.shape)
print(filter_data[filter_data['RM']==8.78])
print(filtered_data[filtered_data['RM']==8.78])
filtered_data.head()
feature = filtered_data.drop('target', axis = 1) 
prices = filtered_data['target']
feature.shape, prices.shape
x_train,x_test,y_train,y_test=train_test_split(feature,prices,random_state=42,test_size=0.2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Calling the LinearRegression and saving it in object
lin_reg = LinearRegression()

#Inbuilt .fit function is used for the training
lin_reg.fit(x_train, y_train)

#Inbuilt .predict function is used for prediction
lin_reg_pred = lin_reg.predict(x_test)

# model evaluation for testing set
#We will evaluate our model using RMSE and R2-score.
acc_rf = r2_score(y_test, lin_reg_pred)
print("The model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(acc_rf))
from sklearn.ensemble import RandomForestRegressor

#Calling the RandomForestRegressor and saving it in object
reg_2 = RandomForestRegressor(max_depth=10,max_leaf_nodes=23, random_state=42)

#Inbuilt .fit function is used for the training
reg_2.fit(x_train, y_train)

#Inbuilt .predict function is used for prediction
reg2_pred = reg_2.predict(x_test)

# model evaluation for testing set
#We will evaluate our model using RMSE and R2-score.
acc_rf = r2_score(y_test, reg2_pred)
%matplotlib inline
plt.scatter(y_test, reg2_pred,color='maroon')
plt.plot([y.min(), y.max()], [y.min(), y.max()], c='yellow', lw=2)
print("The model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(acc_rf))
from sklearn.linear_model import Lasso

#Calling the LassoRegressor and saving it in object
reg_3 = Lasso(random_state=42)

#Inbuilt .fit function is used for the training
reg_3.fit(x_train, y_train)

#Inbuilt .predict function is used for prediction
y_pred = reg_3.predict(x_test)

# model evaluation for testing set
#We will evaluate our model using RMSE and R2-score.
acc_rf = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(acc_rf))
