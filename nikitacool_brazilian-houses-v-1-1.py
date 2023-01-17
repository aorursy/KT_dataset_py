import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv")
data2 = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
# Unexpected "Unnamed: 0" column meaning id column
data
data2
print(data['city'].value_counts(), "\n") # Sao Paulo?
print(data2['city'].value_counts())
# Copy the data
data_city = data
data_city = data.drop('Unnamed: 0', axis=1)

# In order to label encoding of the second dataset I change to another numbers now.
data_city['city'] = data_city['city'].replace(1,4)
data_city['city'] = data_city['city'].replace(0,1)
data_city['hoa (R$)'] = data_city['hoa']
data_city['rent amount (R$)'] = data_city['rent amount']
data_city['property tax (R$)'] = data_city['property tax']
data_city['fire insurance (R$)'] = data_city['fire insurance']

data_city = data_city.drop('total', axis=1)
data_city = data_city.drop('hoa', axis=1)
data_city = data_city.drop('rent amount', axis=1)
data_city = data_city.drop('property tax', axis=1)
data_city = data_city.drop('fire insurance', axis=1)
for i in data_city:
    data_city[i] = data_city[i].replace("-", 1)
def convert_to_num(value):
    num = value.replace('R$', '')
    num = num.replace(',', '')
    num = float(num)
    return num
data_city['hoa (R$)'] = data_city['hoa (R$)'].replace("Sem info", '0')
data_city['hoa (R$)'] = data_city['hoa (R$)'].replace("Incluso", '0')

data_city['property tax (R$)'] = data_city['property tax (R$)'].replace("Sem info", '0')
data_city['property tax (R$)'] = data_city['property tax (R$)'].replace("Incluso", '00')

data_city['hoa (R$)'] = data_city['hoa (R$)'].apply(lambda x: convert_to_num(x))
data_city['rent amount (R$)'] = data_city['rent amount (R$)'].apply(lambda x: convert_to_num(x))
data_city['property tax (R$)'] = data_city['property tax (R$)'].apply(lambda x: convert_to_num(x))
data_city['fire insurance (R$)'] = data_city['fire insurance (R$)'].apply(lambda x: convert_to_num(x))

data_city['animal'] = data_city['animal'].apply(lambda x : 1 if x=='acept' else 0)
data_city['furniture'] = data_city['furniture'].apply(lambda x : 1 if x=='furnished' else 0)

data_city['floor'] = pd.to_numeric(data_city['floor'])
data_['floor'] = pd.to_numeric(data_['floor'])
data_city
data2
# Make a copy of data2
data_city_2 = data2.copy(deep=True)
for i in data_city_2:
    data_city_2[i] = data_city_2[i].replace("-", 1)

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

data_city_2['city'] = label.fit_transform(data_city_2['city'])
data_city['city'].value_counts()
data_city_2['city'].value_counts()
# Add Campinas
Campinas_data = data_city[data_city['city']==1]
data_city_2 = pd.concat([data_city_2, Campinas_data])
data_city_2['animal'] = data_city_2['animal'].apply(lambda x : 1 if x=='acept' else 0)
data_city_2['furniture'] = data_city_2['furniture'].apply(lambda x : 1 if x=='furnished' else 0)
data_ = data_city_2.drop('total (R$)', axis=1)
data_
# import modules
import matplotlib.pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
%pylab inline
plt.figure(figsize =(10,10))
corr = data_.corr()
sns.heatmap(corr, annot =True)
num = 0
num_col = [col for col in data_ if data_[col].dtype != np.dtype(np.object)]

pyplot.figure(figsize=(25,25))

for i in num_col:
    num += 1
    pyplot.subplot(5,4,num)
    pyplot.scatter(data_[i], data_['rent amount (R$)'], s=1)
    pyplot.title(i)
# It seems unrealistic.
data_[data_['hoa (R$)'] > 20000]
# and here
data_[data_['property tax (R$)'] > 20000]
# and here
data_[data_['area'] > 22000]
data_ = data_.drop(data_[data_['property tax (R$)'] > 20000].index, axis=0 )
data_ = data_.drop(data_[data_['hoa (R$)'] > 20000].index, axis=0 )
data_ = data_.drop(data_[data_['area'] > 12000].index, axis=0 )
num = 0
num_col = [col for col in data_ if data_[col].dtype != np.dtype(np.object)]

pyplot.figure(figsize=(25,25))

for i in num_col:
    num += 1
    pyplot.subplot(5,4,num)
    pyplot.scatter(data_[i], data_['rent amount (R$)'], s=1)
    pyplot.title(i)
data_.skew()
test_data = data_city[:5]

predict = test_data.drop(['rent amount (R$)'], axis=1)
actual = test_data['rent amount (R$)']
X = data_.drop(['rent amount (R$)'], axis=1)
y = data_['rent amount (R$)']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae

train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.75, test_size=0.25)

model = LinearRegression()
model.fit(train_X, train_y)
pred = model.predict(valid_X)
mae(pred, valid_y)
from xgboost import XGBRegressor
model2 = XGBRegressor()
model2.fit(train_X, train_y)

pred = model2.predict(valid_X)
mae(pred, valid_y)
final_pred = model2.predict(predict)

print("Predicted/Recommended Rent Price =" ,final_pred)
print("Actual Rent Price =", list(actual))