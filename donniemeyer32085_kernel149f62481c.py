# Uncomment and run cellto get to the data source



import webbrowser

#webbrowser.open('https://www.kaggle.com/slavapasedko/belarus-used-cars-prices')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/belarus-used-cars-prices/cars.csv')
df.info()
df.head()
df.tail()
df['transmission'].unique()
df.describe(include='all')
missing_data = df.isnull().sum() 

missing_data
na_index_filter = missing_data[missing_data / len(df) <= 0.05].index
df1 = df[na_index_filter]

df1.info()
(df1['volume(cm3)'].isnull().sum() / len(df)) * 100
(df1['drive_unit'].isnull().sum() / len(df)) * 100
df1['drive_unit'].unique()
df1 = df1.dropna(axis=0)

df1.info()
df1['year']= df1['year'].astype('object')
text_cols = df1.select_dtypes(include=['object']).columns

text_cols
for cat in text_cols:

    print(cat + ':' + str(len(df1[cat].unique())))
for col in text_cols:

    df1[col]= df1[col].astype('category')
df1.dtypes
for col in text_cols:    

    print(col + ':' + str(df1[col].cat.codes.unique()))
df1.info()
text_cols
df1['mileage(miles)_sqrt'] = np.sqrt(df1['mileage(kilometers)']*0.621371)
df1.head()
num_features = df1.select_dtypes(include=[int, float]).columns.drop('priceUSD')

num_features
CORR_MAP = df1.corr()

CORR_MAP
plt.figure(figsize=(10, 7))

sns.heatmap(CORR_MAP, annot=True)
price_corr = df1.corr()['priceUSD'].sort_values().drop('priceUSD')

price_corr
feature_drop = price_corr[np.abs(price_corr) <= 0.25].index

feature_drop
df1 = df1.drop(feature_drop, axis=1)

df1.head()
cat_features = df1.select_dtypes(include=['category']).columns

cat_features
dummy_cols = pd.DataFrame()

for text_col in text_cols:

    col_dummies = pd.get_dummies(df1[text_col])

    df1 = pd.concat([df1, col_dummies], axis=1)

    del df1[text_col]
df1.head()
corr_price = df1.corr()['priceUSD'].sort_values()
corr_price = corr_price.drop('priceUSD')
other_index = corr_price[np.abs(corr_price) > 0.15].index                            
new_df = df1[other_index]

new_df['priceUSD'] = df1['priceUSD']
new_df.head()
train_rows = len(df1) * .75

test_rows = len(df1) - train_rows

print(train_rows)

print(test_rows)

train_rows + test_rows == len(df1)
train_df = df1.iloc[:40720]

test_df = df1.iloc[40720:]
features = train_df.drop('priceUSD', axis=1).columns

target = 'priceUSD'
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
lm = LinearRegression()
lm.fit(train_df[features], train_df[target])
test_predictions = lm.predict(test_df[features])
actual_target = test_df[target].reset_index(drop=True)

actual_target.head()
pred_df = pd.concat([actual_target, pd.Series(test_predictions)], axis=1, ignore_index=True)

pred_df.columns = ['Actual', 'Predicted']

pred_df.head()
plot_df = pred_df.head(50)

plot_df.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
rmse = np.sqrt(mean_squared_error(test_df[target], test_predictions))

rmse
rmse = np.sqrt(mean_squared_error(test_df[target], test_predictions))

rmse
lm = LinearRegression()
lm.fit(train_df[features], train_df[target])
test_predictions = lm.predict(test_df[features])
actual_target = test_df[target].reset_index(drop=True)

actual_target.head()
pred_df = pd.concat([actual_target, pd.Series(test_predictions)], axis=1, ignore_index=True)

pred_df.columns = ['Actual', 'Predicted']

pred_df.head()
plot_df = pred_df.head(50)

plot_df.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
rmse = np.sqrt(mean_squared_error(test_df[target], test_predictions))

rmse