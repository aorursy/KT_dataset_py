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
data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
data.head()
data.info()
data.describe()
data['ocean_proximity'].value_counts()
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



california_img = mpimg.imread('../input/california-housing-feature-engineering/california.png')

data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(20,14), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)

plt.show()
_ = data.hist(bins=50, figsize=(20,15))
data[data['median_house_value'] >= 500001].count()
data = data[data['median_house_value'] < 500001]
df = data[data.ocean_proximity != 'ISLAND']

df.reset_index()

df.head()
_ = df.hist(bins=50, figsize=(20,15))
c = ["INLAND", "<1H OCEAN", "NEAR BAY", "NEAR OCEAN"]

df = df.assign(**dict.fromkeys(c, 0))

df.head()
for key in ['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN']:

    df.loc[df["ocean_proximity"] == key, key] = 1

df
df = df.drop(columns=['ocean_proximity'])

df.head()
col = df.columns.tolist()

print("Current columns names: \n", col)

new_cols = ['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

df = df.reindex(columns=new_cols)

print("Reordered columns names: \n", df.columns.tolist())
df_NaN = df[pd.isnull(df).any(axis=1)]

print("Number of NaN values:\n", df.isnull().sum(axis = 0))

print(df_NaN.head())



df_NaN.to_csv('df_NaN.csv')

### Note: there 207 missing values in the 'total bedrooms' column
ave_total_rooms = df["total_rooms"].mean()

ave_total_bedrooms = df["total_bedrooms"].mean()

ave_median_income = df["median_income"].mean()



med_total_rooms = df["total_rooms"].median()

med_total_bedrooms = df["total_bedrooms"].median()

med_median_income = df["median_income"].median()
# Case 2: Replace with median values caluclation

df["total_bedrooms"] = df["total_bedrooms"].fillna(med_total_bedrooms / med_total_rooms * df["total_rooms"])

print("Number of NaN values after filling:\n", df.isnull().sum(axis = 0))

df.to_csv('df_filled.csv')
df.head()
# clean_data = data.dropna()
# clean_data.head()
# clean_data.drop('ocean_proximity', axis='columns', inplace=True)
# clean_data.head()
df["total_bedrooms"] = df["total_bedrooms"] / df['households']

df["total_rooms"] = df["total_rooms"] / df['households']

df.head()
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(df)

df_origin = df

df.loc[:,:] = scaled_values

df.to_csv('df_scaled.csv')

df.head()
from sklearn.preprocessing import StandardScaler



df = df_origin

scaler = StandardScaler() 

scaled_df = scaler.fit_transform(df) 

df.loc[:,:] = scaled_values

df.head()
from sklearn.model_selection import train_test_split



train, test = train_test_split(df, test_size=0.25, random_state=42)
train.info(), test.info()
col = test.columns.tolist()

col
features = col[: -1]

target = col[-1]

features
X_train = train[features]

y_train = train[target]



X_test = test[features]

y_test = test[target]
from sklearn.linear_model import LinearRegression



model = LinearRegression()
_ = model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error



mean_squared_error(y_test, predictions)
from sklearn.metrics import r2_score



r2_score(y_test, predictions)
model.score(X_test, y_test)