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
data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

data.head()
data_NEARBAY = data[data['ocean_proximity']=="NEAR OCEAN"]
resultat = data.groupby('ocean_proximity')['median_house_value'].nunique()
print(resultat)
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 10))
x = data['median_income']
y = data['median_house_value']
plt.scatter(x, y, marker='o')
x1 = data_NEARBAY['median_income']
y1 = data_NEARBAY['median_house_value']
plt.scatter(x1, y1, marker='+', color = 'red')
plt.title("Variation in House prices")
plt.xlabel('median_income')
plt.ylabel('"House prices in $1000"')
plt.show()
data = data_NEARBAY
data.info()
data.describe()
data['ocean_proximity'].value_counts()
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# california_img = mpimg.imread('../input/california-housing-feature-engineering/california.png')
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.show()
_ = data.hist(bins=50, figsize=(20,15))
data[data['median_house_value'] >= 500001].count()
data[data['housing_median_age'] >= 50].count()
plt.figure(figsize=(30, 10))
features =[]
features.extend(data.columns)
features.remove('median_house_value')
print (features)
target = data['median_house_value']


for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = data[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')
# Capped variables
clean_data = data[data['median_house_value'] <= 499999]
clean_data = clean_data[clean_data['housing_median_age'] <= 50]

# Missing values
clean_data = clean_data.dropna()

# Categorical variable: ocean proximity (encode it)

# clean_data["ocean_proximity"]= clean_data["ocean_proximity"].astype('category')
# clean_data["ocean_proximity_cat"]= clean_data["ocean_proximity"].cat.codes
clean_data.drop(['ocean_proximity'], axis =1)
clean_data.head()

# Feature Engineering

clean_data['RoomsByHouseholds'] = clean_data['total_rooms']/clean_data['households']
clean_data['BedroomsByHouseholds'] = clean_data['total_bedrooms']/clean_data['households']

# Features
features =[]
features.extend(clean_data.columns)
features.remove('ocean_proximity')
target = 'median_house_value'
features.remove('median_house_value')
print ("feature {}".format(features))
Y = clean_data[target]
X = clean_data[features]
X.head()
from sklearn.model_selection import train_test_split
import numpy as np
Xtrain, Xtest, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42)
name_column = Xtrain.columns
y_train = np.array(y_train).reshape(len(y_train),1)
y_test = np.array(y_test).reshape(len(y_test),1)
from sklearn import preprocessing
scalerX = preprocessing.StandardScaler().fit(Xtrain)
scalery = preprocessing.StandardScaler().fit(y_train)
Xtrain_scaled = scalerX.transform(Xtrain)
Xtest_scaled = scalerX.transform(Xtest)
y_train_scaled = scalery.transform(y_train)
y_test_scaled = scalery.transform(y_test)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
_ = model.fit(Xtrain, y_train)
y_pred_test = np.array(model.predict(Xtest)).reshape(len(model.predict(Xtest)),1)
y_test_pred_Unnorm = scalery.inverse_transform(y_pred_test)
y_pred_test[0:5]
print("Maximum house price : {} $".format(int(max(y_test))))
print("Minimum and Maximum predicted house price : {} $ and {} $".format(int(min(y_pred_test)), int(max(y_pred_test))))
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)
fig, ax = plt.subplots(constrained_layout=False)
x = y_test/1000.
y = y_test_pred_Unnorm/1000.
ax.scatter(x, y, marker = "o")
# ax.yaxis.set_ticks(range(500,50))
ax.yaxis.grid(True, color = 'orange', linewidth = 1, linestyle = 'dashed')
ax.set_xlabel("Actual House Prices ($1000)")
ax.set_ylabel("Predicted House Prices: ($1000)")
ax.set_title("Actual Prices vs Predicted prices")
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred_test)
from sklearn.metrics import r2_score

r2_score( y_test, y_pred_test)
model.score(Xtest, y_test)