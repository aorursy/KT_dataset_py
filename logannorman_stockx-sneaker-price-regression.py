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

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as plticker

%matplotlib inline
# Reading in the data

original_data = pd.read_csv('../input/stockx/StockX-Data-Contest-2019-3.csv', header = 2)

df = original_data.copy()

df.head()
df.shape
df.dtypes
df.describe()
# Checking for null values

nulls = pd.concat([df.isnull().sum()], axis=1)

nulls[nulls.sum(axis=1) > 0]
# Change 'order date' dtype

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')

df.head()
# Change 'release date' dtype

df['Release Date'] = pd.to_datetime(df['Release Date'], format='%m/%d/%Y')

df.head()
# Remove - from sneaker name

df['Sneaker Name'] = df['Sneaker Name'].apply(lambda x: x.replace('-', ' '))

df.head()
# Remove $ and comma from sale price

df['Sale Price'] = df['Sale Price'].apply(lambda x: x.replace('$', ''))

df['Sale Price'] = df['Sale Price'].apply(lambda x: x.replace(',', ''))

df.head()
# Remove $ from retail price

df['Retail Price'] = df['Retail Price'].apply(lambda x: x.replace('$', ''))

df.head()
df.dtypes
# Converting some object columns into numerical columns

obj_cols = ['Sale Price','Retail Price']

for col in obj_cols:

    df[str(col)] = pd.to_numeric(df[str(col)])
df.dtypes
df.columns
# Make Bought For Less Than Retail column

df['Bought for Less Than Retail'] = df['Sale Price'] < df['Retail Price']

df.head()
# Make Bought For Retail column

df['Bought for Retail'] = df['Sale Price'] == df['Retail Price']

df.head()
# Make Bought For More Than Retail column

df['Bought for More Than Retail'] = df['Sale Price'] > df['Retail Price']

df.head()
# Genereal numeric correlations

# Analyze trend between shoe size and sale price

# Analyze trend between sale price and retail price

correlations = df.corr()

sns.heatmap(correlations)
# Release date, buyer region, and sneaker name, retail price, shoe size, and brand distribution analysis

df_cat = ['Release Date', 'Buyer Region', 'Sneaker Name', 'Retail Price', 'Shoe Size', 'Brand', 'Bought for Retail', 'Bought for Less Than Retail', 'Bought for More Than Retail' ]

for cat in df_cat:

    cat_num = df[str(cat)].value_counts()

    plt.figure(figsize=(15,6))

    chart = sns.barplot(x = cat_num.index, y= cat_num)

    chart.set_title("Sneakers Sales by %s" % (cat))

    plt.ylabel("Sneaker Sales")

    chart.set_xticklabels(chart.get_xticklabels(), rotation = 90)

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    plt.show(15)
# Analyze trend between Sneaker Name & Sale price

# Create average retail price df

bruh = df[['Sneaker Name', 'Sale Price']]



# Clean up this list

sneakernames = ['adidas Yeezy Boost 350 V2 Butter',

       'Adidas Yeezy Boost 350 V2 Beluga 2pt0',

       'Adidas Yeezy Boost 350 V2 Zebra',

       'Adidas Yeezy Boost 350 V2 Blue Tint',

       'Adidas Yeezy Boost 350 V2 Cream White',

       'Adidas Yeezy Boost 350 V2 Sesame', 'adidas Yeezy Boost 350 V2 Static',

       'Adidas Yeezy Boost 350 V2 Semi Frozen Yellow',

       'Air Jordan 1 Retro High Off White University Blue',

       'adidas Yeezy Boost 350 V2 Static Reflective',

       'Nike Air Presto Off White Black 2018',

       'Nike Air Presto Off White White 2018',

       'Nike Air VaporMax Off White 2018',

       'Nike Blazer Mid Off White All Hallows Eve',

       'Nike Blazer Mid Off White Grim Reaper', 'Nike Zoom Fly Off White Pink',

       'Nike Air VaporMax Off White Black',

       'Nike Zoom Fly Off White Black Silver',

       'Nike Air Force 1 Low Off White Volt',

       'Adidas Yeezy Boost 350 V2 Core Black Red 2017',

       'Nike Air Force 1 Low Off White Black White',

       'Air Jordan 1 Retro High Off White Chicago',

       'Nike Air Max 90 Off White Black',

       'Nike Zoom Fly Mercurial Off White Total Orange',

       'Nike Air Max 90 Off White Desert Ore',

       'Nike Zoom Fly Mercurial Off White Black', 'Nike Air Max 90 Off White',

       'Adidas Yeezy Boost 350 V2 Core Black White',

       'Nike Air Presto Off White', 'Nike Air Max 97 Off White',

       'Nike Air VaporMax Off White', 'Nike Blazer Mid Off White',

       'Adidas Yeezy Boost 350 Low V2 Beluga',

       'Nike React Hyperdunk 2017 Flyknit Off White',

       'Nike Air Force 1 Low Off White', 'Nike Zoom Fly Off White',

       'Nike Air Max 97 Off White Menta',

       'Air Jordan 1 Retro High Off White White',

       'Adidas Yeezy Boost 350 V2 Core Black Red',

       'Nike Air Max 97 Off White Black',

       'Nike Blazer Mid Off White Wolf Grey',

       'Adidas Yeezy Boost 350 V2 Core Black Copper',

       'Nike Air Max 97 Off White Elemental Rose Queen',

       'Adidas Yeezy Boost 350 V2 Core Black Green',

       'Adidas Yeezy Boost 350 Low Pirate Black 2016',

       'Adidas Yeezy Boost 350 Low Moonrock',

       'Adidas Yeezy Boost 350 Low Pirate Black 2015',

       'Adidas Yeezy Boost 350 Low Oxford Tan',

       'Adidas Yeezy Boost 350 Low Turtledove',

       'Nike Air Force 1 Low Virgil Abloh Off White AF100'

       ]

avgs = []

for name in sneakernames:

    shoerow = bruh.loc[bruh['Sneaker Name'] == name]

    avgs.append(shoerow.mean()[0])

AvgPrice = pd.Series(avgs)

SneakerName = pd.Series(sneakernames)

avgprice_df = pd.DataFrame(columns = ['Sneaker_Name', 'Average_Price'])

avgprice_df['Sneaker_Name'] = SneakerName

avgprice_df['Average_Price'] = AvgPrice



# Crerating visual of average shoe price

fig_dims = (15, 4)

fig, ax = plt.subplots(figsize=fig_dims)

chart = sns.barplot(x = avgprice_df['Sneaker_Name'] , y= avgprice_df['Average_Price'])

chart.set_xticklabels(chart.get_xticklabels(), rotation = 90)

plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
# Find average sale price by brand

avgs_2 = []

bds = df[['Brand', 'Sale Price']]

brands = [' Yeezy', 'Off-White']

for brand in brands:

    brandrow = bds.loc[bds['Brand'] == str(brand)]

    avgs_2.append(brandrow['Sale Price'].mean())

print('Yeezy average price: $' + str(avgs_2[0]))

print('Off-White average price: $' + str(avgs_2[1]))
# Create boxplot distribution of sales price by brand

for brand in brands:

    brandrow = bds.loc[bds['Brand'] == str(brand)]

    chart = sns.boxplot(y=brandrow["Sale Price"], showfliers = False)

    chart.set_title("Sale Price Distribution of %s sneakers" % (brand))

    plt.show()
# Analyze trend between Sales Price and Order Date

# Find average sale price per day

dts = df[['Order Date', 'Sale Price']]

uniq_ord_dates = df['Order Date'].value_counts().index.tolist()

avg_3 = []



for date in uniq_ord_dates:

    daterow = dts.loc[dts['Order Date'] == str(date)]

    avg_3.append(daterow['Sale Price'].mean())



unq_dates = pd.Series(uniq_ord_dates)

date_avgs = pd.Series(avg_3)

dateprice_df = pd.DataFrame(columns = ['Order_date', 'Average_Price'])

dateprice_df['Order_date'] = unq_dates.sort_values(ascending = True)

dateprice_df['Average_Price'] = date_avgs

dateprice_df.head()
# Create visualization of Average Sale Price Over time

fig_dims = (20, 5)

fig, ax = plt.subplots(figsize=fig_dims)

chart = sns.scatterplot(x="Order_date", y="Average_Price", data=dateprice_df)

plt.gca().xaxis.set_major_locator(plt.MultipleLocator(40))

chart.set_title("Average Daily Sale Price Over time")
# Finding Average Sale Price on Release Dates Over Time

dts = df[['Release Date', 'Sale Price']]

uniq_rel_dates = df['Release Date'].value_counts().index.tolist()

avg_4 = []



for date in uniq_rel_dates:

    daterow = dts.loc[dts['Release Date'] == str(date)]

    avg_4.append(daterow['Sale Price'].mean())



unq_dates = pd.Series(uniq_rel_dates)

date_avgs = pd.Series(avg_4)

dateprice_df_2 = pd.DataFrame(columns = ['Release_date', 'Average_Price'])

dateprice_df_2['Release_date'] = unq_dates.sort_values(ascending = True)

dateprice_df_2['Average_Price'] = date_avgs

dateprice_df_2.head()
# Analyze trend between buyer region and sale price



brg = df[['Buyer Region', 'Sale Price']]

unq_brgs = df['Buyer Region'].value_counts().index.tolist()

avg_5 = []



for region in unq_brgs:

    regionrow = brg.loc[brg['Buyer Region'] == str(region)]

    avg_5.append(regionrow['Sale Price'].mean())



unq_regions = pd.Series(unq_brgs)

region_avgs = pd.Series(avg_5)

regionprice_df = pd.DataFrame(columns = ['Buyer Region', 'Average Price'])

regionprice_df['Buyer Region'] = unq_regions.sort_values(ascending = True)

regionprice_df['Average Price'] = region_avgs



fig_dims = (11, 10)

fig, ax = plt.subplots(figsize=fig_dims)

chart = sns.barplot(x="Average Price", y="Buyer Region", data=regionprice_df, color="b")

plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))

chart.set_title("Average Sale Price by Buyer Region")
# Renaming columns to get rid of spaces 

df = df.rename(columns={

    "Order Date": "Order_date",

    "Sneaker Name": "Sneaker_Name",

    "Sale Price": "Sale_Price",

    "Retail Price": "Retail_Price",

    "Release Date": "Release_Date",

    "Shoe Size": "Shoe_Size",

    "Buyer Region": "Buyer_Region"

    })
# Converting dates into numericals

import datetime as dt



df['Order_date'] = pd.to_datetime(df['Order_date'])

df['Order_date']=df['Order_date'].map(dt.datetime.toordinal)



df['Release_Date'] = pd.to_datetime(df['Release_Date'])

df['Release_Date']=df['Release_Date'].map(dt.datetime.toordinal)
# Getting spltis

from sklearn import preprocessing, metrics

from sklearn.model_selection import train_test_split



X = df.drop(['Sale_Price', 'Bought for More Than Retail', 'Bought for Less Than Retail', 'Bought for Retail'], axis=1)

y = df.Sale_Price

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
# Converting categorical data to numerical

from sklearn.preprocessing import OneHotEncoder



object_cols = ['Sneaker_Name', 'Buyer_Region', 'Brand']

# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Adding the column names after one hot encoding

OH_cols_train.columns = OH_encoder.get_feature_names(object_cols)

OH_cols_valid.columns = OH_encoder.get_feature_names(object_cols)



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
# Starting linear regression

from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(OH_X_train,y_train)
# Looking at y-int

print(lm.intercept_)
# Looking at coefficient scores of each variable

coeff_df = pd.DataFrame(lm.coef_, OH_X_train.columns,columns=['Coefficient'])

ranked_coeff = coeff_df.sort_values("Coefficient", ascending = False)

ranked_coeff
# Storing predictions and running evaluation metrics

predictions = lm.predict(OH_X_valid)

from sklearn import metrics

print("MAE:", metrics.mean_absolute_error(y_valid, predictions))

print('MSE:', metrics.mean_squared_error(y_valid, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()

lm.fit(OH_X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(OH_X_train, y_train)
list(zip(OH_X_train.columns,rfe.support_,rfe.ranking_))
X_train_rfe = OH_X_train[OH_X_train.columns[rfe.support_]]

X_train_rfe
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    model = sm.OLS(y, X)

    results = model.fit() # fitting the model

    print(results.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_new = build_model(OH_X_train,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(['Sneaker_Name_Adidas Yeezy Boost 350 V2 Core Black Green',

       'Sneaker_Name_Nike Air Force 1 Low Off White',

       'Sneaker_Name_Nike Air Max 90 Off White',

       'Sneaker_Name_Nike Air VaporMax Off White Black',

       'Buyer_Region_Alabama', 'Buyer_Region_Alaska', 'Buyer_Region_Arkansas',

       'Buyer_Region_Colorado', 'Buyer_Region_Connecticut',

       'Buyer_Region_Delaware', 'Buyer_Region_District of Columbia',

       'Buyer_Region_Georgia', 'Buyer_Region_Hawaii', 'Buyer_Region_Idaho',

       'Buyer_Region_Illinois', 'Buyer_Region_Indiana', 'Buyer_Region_Iowa',

       'Buyer_Region_Kansas', 'Buyer_Region_Louisiana', 'Buyer_Region_Maine',

       'Buyer_Region_Massachusetts', 'Buyer_Region_Michigan',

       'Buyer_Region_Minnesota', 'Buyer_Region_Mississippi',

       'Buyer_Region_Missouri', 'Buyer_Region_Montana',

       'Buyer_Region_Nebraska', 'Buyer_Region_Nevada',

       'Buyer_Region_New Hampshire', 'Buyer_Region_New Jersey',

       'Buyer_Region_New Mexico', 'Buyer_Region_New York',

       'Buyer_Region_North Carolina', 'Buyer_Region_North Dakota',

       'Buyer_Region_Ohio', 'Buyer_Region_Oklahoma',

       'Buyer_Region_Pennsylvania', 'Buyer_Region_Rhode Island',

       'Buyer_Region_South Carolina', 'Buyer_Region_South Dakota',

       'Buyer_Region_Tennessee', 'Buyer_Region_Texas', 'Buyer_Region_Utah',

       'Buyer_Region_Vermont', 'Buyer_Region_Virginia',

       'Buyer_Region_Washington', 'Buyer_Region_West Virginia',

       'Buyer_Region_Wyoming'], axis=1)
X_train_new = build_model(X_train_new,y_train)
# Dropping brand names because of collinearity

X_train_new = X_train_new.drop(['Brand_Off-White', 'Brand_ Yeezy'], axis=1)
best_X_train = build_model(X_train_new,y_train)
bruv = checkVIF(best_X_train)

bruv
lm = sm.OLS(y_train, best_X_train).fit()

y_train_price = lm.predict(best_X_train)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 50)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)
#Dividing into X and y

y_test = y_valid

X_test = OH_X_valid
# Now let's use our model to make predictions.

X_train_new = best_X_train.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
# Scoring the model

from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)
print(lm.summary())