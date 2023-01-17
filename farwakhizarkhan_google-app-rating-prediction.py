# Following Libraries are being used

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import linear_model #For missing values

from sklearn.preprocessing import StandardScaler #For scaling features



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor #Random Forest Regressor

from sklearn.ensemble import BaggingRegressor #Bagging Regressor

from sklearn.neighbors import KNeighborsRegressor #KNN Regressor

from sklearn.linear_model import LinearRegression #Linear Regressor
# Load training data

data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

data.head()
# Check dimensions of the data

data.shape
# Explore columns

data.columns
# Description

data.describe()
# Check Datatypes

data.dtypes
# Drop the columns which have no impact

data = data.drop(columns=['App', 'Last Updated', 'Current Ver', 'Android Ver'])
# Check missing values in each column of training data

data.isnull().sum()
# Check if any of the record has rating > 5

data[data['Rating'] > 5]
# Now check if 1.9 is the real category or its a dummy data

data['Category'].unique().tolist()
# So its obvious that 1.9 category and rating 

# above 5 doesn't make sense, so drop this record

rec = data[data['Category'] == '1.9'].index

data = data.drop(rec)
# Make a copy of data

train_data = data.copy()
# Cleaning "Price" column

train_data['Price'] = train_data['Price'].apply(lambda x : x.strip('$'))

# Cleaning "Installs" column

train_data['Installs'] = train_data['Installs'].apply(lambda x : x.strip('+').replace(',', ''))
# Convert all these to float

train_data['Price'] = train_data['Price'].astype(float)

train_data['Installs'] = train_data['Installs'].astype(float)

train_data['Reviews'] = train_data['Reviews'].astype(float)
# Remove record with Type nan

rec = train_data[train_data['Type'].isnull()].index

train_data = train_data.drop(rec)
# Get all features with type 'object'

col_list = [c for c in train_data.columns if train_data[c].dtype == 'object']

col_list
# Encode features except Size

for c in col_list:

    if c != 'Size':

        train_data[c] = train_data[c].astype('category')

        train_data[c] = train_data[c].cat.codes
# Compare actual and encoded labels for column 'Type'

print(data['Type'].unique().tolist())

print(train_data['Type'].unique().tolist(), '\n')
# Replace "Varies with device" in Size with null value

train_data.loc[train_data['Size'] == 'Varies with device', 'Size'] = np.nan
# Removing the suffixes (k and M) and representing all the data as bytes 

# (i.e)for k, value is multiplied by 1000 and for M, the value is multiplied by 1000000 

train_data.Size = (train_data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \

             train_data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1)

            .replace(['k','M'], [10**3, 10**6]).astype(int))
# Now check null values

train_data.isnull().sum()
# For Size missing values

# Get rows which are not null for Size

X = train_data[train_data['Size'].notnull()]

y = train_data.loc[train_data['Size'].notnull(), 'Size']

X = X.drop(columns=['Size', 'Rating'])



# Fit the model

model = linear_model.LinearRegression()

model.fit(X, y)



# Get all rows with null values

X_miss = train_data[train_data['Size'].isnull()]

X_miss = X_miss.drop(columns = ['Size', 'Rating'])



# Fill the predicted values

train_data.loc[train_data['Size'].isnull(), 'Size'] = model.predict(X_miss)
# For Rating missing values

X = train_data[train_data['Rating'].notnull()]

y = train_data.loc[train_data['Rating'].notnull(), 'Rating']

X = X.drop(columns=['Rating'])



# Fit model

model = linear_model.LinearRegression()

model.fit(X, y)



# Get all rows with null values

X_miss = train_data[train_data['Rating'].isnull()]

X_miss = X_miss.drop(columns = ['Rating'])



# Fill the predicted values

train_data.loc[train_data['Rating'].isnull(), 'Rating'] = model.predict(X_miss)
train_data.isnull().sum()
# Final data type of the data

train_data.dtypes
#Analyse the preprocessed data

train_data.head()
# Correlation heatmap

corr = train_data.corr() 

plt.figure(figsize=(9, 8))



sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True)
from pylab import rcParams

import warnings

warnings.filterwarnings('ignore')



# rating distibution 

rcParams['figure.figsize'] = 11,8

g = sns.kdeplot(data.Rating, color="Red", shade = True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

plt.title('Distribution of Rating',size = 20)
#Game and Family category are the most appearances for application in store

plt.figure(figsize=(10, 5))

g = sns.countplot(x="Category",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Count of app in each category')
# about 93% of the apps are free on google playstore

labels =data['Type'].value_counts(sort = True).index

sizes = data['Type'].value_counts(sort = True)





colors = ["lightblue","orange"]

explode = (0.1,0)  # explode 1st slice

 

rcParams['figure.figsize'] = 7,7

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('Percent of Free and paid Apps in store',size = 20)

plt.show()
# Standardize data

# Columns not to be standardized. These are columns with categorical data, 

# also we don't standardize our target vraiable

cols = ['Category', 'Type', 'Content Rating', 'Genres', 'Rating']



# Pick remaining columns and standardize them 

columns = [c for c in train_data.columns if c not in cols]

scaler = StandardScaler()

scaler.fit(train_data[columns])

train_data[columns] = scaler.transform(train_data[columns])



# Check data after standarization

train_data.head()
# Train Test Split

# Split data to 80% of the training and 20% for the validation

y = train_data['Rating']

X = train_data.drop(columns=['Rating'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training Set Dimensions:", X_train.shape)

print("Validation Set Dimensions:", X_test.shape)
# Train Random Forest Regressor

randomf = RandomForestRegressor(n_estimators=300)

randomf.fit(X_train, y_train)



# Measure mean squared error for training and validation sets

print('Mean squared Error for Training Set:', mean_squared_error(y_train, randomf.predict(X_train)))

print('Mean squared Error for Test Set:', mean_squared_error(y_test, randomf.predict(X_test)))
# Important features for random forest regressor

for name, importance in zip(X.columns, randomf.feature_importances_):

    print('feature:', name, "=", importance)

    

importances = randomf.feature_importances_

indices = np.argsort(importances)

features = X.columns

plt.figure(figsize=(6, 4))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='g', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
# Fit model

br = BaggingRegressor(random_state=300)

                            

br.fit(X_train, y_train)



# Measure mean squared error for training and validation sets

print('Mean squared Error for Training Set:', mean_squared_error(y_train, br.predict(X_train)))

print('Mean squared Error for Test Set:', mean_squared_error(y_test, br.predict(X_test)))
# Fit model

knr = KNeighborsRegressor(n_neighbors = 5)

knr.fit(X_train, y_train)



# Measure mean squared error for training and validation sets

print('Mean squared Error for Training Set:', mean_squared_error(y_train, knr.predict(X_train)))

print('Mean squared Error for Test Set:', mean_squared_error(y_test, knr.predict(X_test)))
# Fit the model

model = LinearRegression()

model.fit(X_train, y_train)



# Measure mean squared error for training and validation sets

print('Mean squared Error for Training Set:', mean_squared_error(y_train, model.predict(X_train)))

print('Mean squared Error for Test Set:', mean_squared_error(y_test, model.predict(X_test)))