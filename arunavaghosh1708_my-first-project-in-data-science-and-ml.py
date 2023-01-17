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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.linear_model import LinearRegression

%matplotlib inline
data=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

data
print(data.shape)

data.head()
data.describe()
data.isnull().sum()
data = data.dropna(subset = ["total_bedrooms"] , axis = 0)

data.shape
data.info()
data.describe()
from pandas import DataFrame
df = DataFrame(data,columns=['latitude','longitude'])

df.plot(x ='latitude', y='longitude',kind = 'scatter')

plt.show()
sns.distplot(data['housing_median_age'],color="red")
house=['NEAR BAY','<1H OCEAN','INLAND','NEAR OCEAN','ISLAND']

house_count=[2270,9034,6496,2628,5]

fig = plt.figure(figsize =(5,5))

explode = (0.1, 0.0, 0.2, 0.2, 0.0)

plt.pie(house_count, labels = house,autopct ='% 1.1f %%',shadow=True,explode=explode)  

plt.show() 
op = data[["ocean_proximity"]]

op["count"] = op.groupby(op.ocean_proximity)["ocean_proximity"].transform("count")

op = op.drop_duplicates()
op
op.plot.bar(x="ocean_proximity", y="count", rot=70, title="Number of houses with respect to ocean");

plt.show()
df1 = DataFrame(data,columns=['total_rooms','median_house_value'])

df1.plot(x ='total_rooms', y='median_house_value',kind = 'scatter',color="orange")

plt.show()
data.columns
min_income_boundary=data["median_income"].quantile (0.01)

max_income_boundary=data["median_income"].quantile (0.99)

min_income_boundary,max_income_boundary
corrected_data = data[(data.median_income<max_income_boundary) & (data.median_income>min_income_boundary)]
min_bedrooms_boundary=corrected_data["total_bedrooms"].quantile (0.01)

max_bedrooms_boundary=corrected_data["total_bedrooms"].quantile (0.99)

min_bedrooms_boundary,max_bedrooms_boundary

corrected_data = corrected_data[(corrected_data.total_bedrooms<max_bedrooms_boundary) & (corrected_data.total_bedrooms>min_bedrooms_boundary)]

corrected_data.shape
min_bedrooms_boundary,max_bedrooms_boundary
min_rooms_boundary=corrected_data["total_rooms"].quantile (0.01)

max_rooms_boundary=corrected_data["total_rooms"].quantile (0.99)

min_rooms_boundary,max_rooms_boundary

corrected_data = corrected_data[(corrected_data.total_rooms<max_rooms_boundary) & (corrected_data.total_rooms>min_rooms_boundary)]

corrected_data.shape
min_rooms_boundary,max_rooms_boundary
min_house_value_boundary=corrected_data["median_house_value"].quantile (0.01)

max_house_value_boundary=corrected_data["median_house_value"].quantile (0.99)

corrected_data = corrected_data[(corrected_data.median_house_value<max_house_value_boundary) & (corrected_data.median_house_value>min_house_value_boundary)]

corrected_data.shape
corrected_data.describe(include='all')
corrected_data.head()
sns.distplot(corrected_data['median_house_value'])
sns.distplot(corrected_data['median_income'])
sns.distplot(corrected_data['households'])
sns.distplot(corrected_data['total_bedrooms'])
sns.distplot(corrected_data['total_rooms'])
plt.figure(figsize = (10 , 8))

sns.heatmap(corrected_data.corr() , annot = True)
corrected_data.head()
corrected_data['ocean_proximity'].value_counts()
corrected_data["ocean_proximity"] = corrected_data["ocean_proximity"].replace({"ISLAND" : 1 ,

                                                       "NEAR BAY" : 2 , 

                                                       "NEAR OCEAN" : 3 , 

                                                       "INLAND" : 4 , 

                                                       "<1H OCEAN" : 5})

corrected_data["ocean_proximity"].value_counts()
corrected_data.describe(include='all')
corrected_data.head()
corrected_data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(corrected_data[["median_house_value"]])
scaled_data=scaler.transform(corrected_data)

scaled_data = pd.DataFrame(scaled_data , columns = corrected_data.columns)

scaled_data.head()
X = scaled_data.drop(["median_house_value"] , axis = 1)

y = scaled_data["median_house_value"]



X.shape , y.shape
from sklearn.model_selection import train_test_split 



x_train , x_test , y_train , y_test = train_test_split(X , y , test_size =0.20)

x_train.shape , x_test.shape , y_train.shape , y_test.shape
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)

y_train.max(),y_train.min(),y_hat.max(),y_hat.min()
plt.scatter(y_train, y_hat)

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

plt.xlim(-1,1.5)

plt.ylim(-0.7,1.06)

plt.show()
sns.distplot(y_train - y_hat)

plt.title("Residuals PDF", size=18)
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(X.columns.values, columns=['Features'])

reg_summary['Weights'] = reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test,alpha=0.2)

plt.xlabel('Targets (y_train)',size=18)

plt.ylabel('Predictions (y_hat)',size=18)

plt.xlim(-1,1.5)

plt.ylim(-0.7,1.06)

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

df_pf.describe()
df_pf.head()
df_pf
df_pf['Target'] = np.exp(y_test)

df_pf
y_test = y_test.reset_index(drop=True)

y_test.head()
df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
df_pf.describe()
pd.options.display.max_rows

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_pf.sort_values(by=['Difference%'])