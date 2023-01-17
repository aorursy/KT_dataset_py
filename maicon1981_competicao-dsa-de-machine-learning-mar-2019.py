import os

path = os.getcwd()
print(os.listdir('../input'))
# Pandas and numpy for data manipulation

import pandas as pd

import numpy as np



# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None



# Matplotlib visualization

import matplotlib.pyplot as plt

%matplotlib inline



# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize



# Seaborn for visualization

import seaborn as sns



# Splitting data into training and testing

from sklearn.model_selection import train_test_split



# Imputing  values and scaling values

from sklearn.preprocessing import MinMaxScaler



# Encode categorical integer features

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Create Deep Neural Network Architecuture

from keras import backend as K

from keras import Sequential

from keras.layers import Dense, Dropout



import warnings

warnings.filterwarnings("ignore")
# Read in data into a dataframe 

train = pd.read_csv('../input/dataset_treino.csv', sep=',')

test  = pd.read_csv('../input/dataset_teste.csv', sep=',')

store = pd.read_csv('../input/lojas.csv', sep=',')
# merge data with store 

train = pd.merge(train, store, on='Store')

test = pd.merge(test, store, on='Store')

del store
# Show dataframe columns

print(train.columns)
# Display top of dataframe

train.head()
# Display bottom of dataframe

train.tail()
# Display the shape of dataframe

train.shape
# See the column data types and non-missing values

train.info()
# Verify missing values

train.isnull().any()
# % of missing values

round((train.isnull().sum() / train.shape[0] * 100), 2)
# Statistics for each column

train.describe()
# Plot histogram

train.hist(figsize=(20,10))
# Boxplots for each column

train.plot(kind='box', subplots=True, layout=(5,3), figsize=(14,10))
print("Distinct number of Stores :", len(train.Store.unique()))

print("Distinct number of Days :", len(train.Date.unique()))

print("Average daily sales of all stores : ", round(train.Sales.mean(),2))
# Count values of DayOfWeek column

train.DayOfWeek.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('Open')

plt.ylabel('Count')

plt.show()
# Check stores open distribution on days of week

sns.countplot(data = train, x = 'DayOfWeek', hue = 'Open')

plt.title('Store Daily Open')
# Count registers with sales equal to zero

np.sum([train.Sales == 0])
# Count values of Open column

train.Open.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('Open')

plt.ylabel('Count')

plt.show()
# Count values of SchoolHoliday column

train.SchoolHoliday.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('SchoolHoliday')

plt.ylabel('Count')

plt.show()
# Count values of StateHoliday column

train.StateHoliday.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('StateHoliday')

plt.ylabel('Count')

plt.show()
# Count values of CompetitionOpenSinceYear column

test.CompetitionOpenSinceYear.value_counts()
# Count values of CompetitionOpenSinceYear column

train.CompetitionOpenSinceYear.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('CompetitionOpenSinceYear')

plt.ylabel('Count')

plt.show()
# Count values of StoreType column

train.StoreType.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('StoreType')

plt.ylabel('Count')

plt.show()
# Count values of Assortment column

train.Assortment.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('Assortment')

plt.ylabel('Count')

plt.show()
# Count values of PromoInterval column

train.PromoInterval.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('PromoInterval')

plt.ylabel('Count')

plt.show()
# Create a histogram to study the Daily Sales for the stores

plt.figure(figsize=(15,8)) 

plt.hist(train.Sales, color='steelblue')  

plt.title("Histogram for Store Sales")

plt.xlabel("bins")

plt.xlabel("Frequency")

plt.show()
# Create the bar plot for Average Sales across different Assortments

ax = sns.barplot(data=train, x='Assortment', y='Sales', color='steelblue') 
# Create the bar plot for Average Sales across different Store Types

ax = sns.barplot(data=train, x='StoreType', y='Sales', color='steelblue') 
# Show dataframe columns

print(test.columns)
# Display top of dataframe

test.head()
# Display bottom of dataframe

test.tail()
# Display the shape of dataframe

test.shape
# See the column data types and non-missing values

test.info()
# Verify missing values

test.isnull().any()
# % of missing values

round((test.isnull().sum() / test.shape[0] * 100), 2)
# Statistics for each column

test.describe()
# Plot histogram

test.hist(figsize=(20,10))
# Boxplots for each column

test.plot(kind='box', subplots=True, layout=(4,3), figsize=(14,10))
print("Distinct number of Stores :", len(test.Store.unique()))

print("Distinct number of Days :", len(test.Date.unique()))
# Count values of DayOfWeek column

test.DayOfWeek.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('Open')

plt.ylabel('Count')

plt.show()
# check stores open distribution on days of week

sns.countplot(data = test, x = 'DayOfWeek', hue = 'Open')

plt.title('Store Daily Open')
# Count values of Open column

test.Open.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('Open')

plt.ylabel('Count')

plt.show()
# Count values of SchoolHoliday column

test.SchoolHoliday.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('SchoolHoliday')

plt.ylabel('Count')

plt.show()
# Count values of StateHoliday column

test.StateHoliday.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('StateHoliday')

plt.ylabel('Count')

plt.show()
# Count values of CompetitionOpenSinceYear column

test.CompetitionOpenSinceYear.value_counts()
# Count values of CompetitionOpenSinceYear column

test.CompetitionOpenSinceYear.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('CompetitionOpenSinceYear')

plt.ylabel('Count')

plt.show()
# Count values of StoreType column

test.StoreType.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('StoreType')

plt.ylabel('Count')

plt.show()
# Count values of Assortment column

test.Assortment.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('Assortment')

plt.ylabel('Count')

plt.show()
# Count values of PromoInterval column

test.PromoInterval.value_counts().plot(kind='bar', figsize=(6,6), color='steelblue')

plt.xlabel('PromoInterval')

plt.ylabel('Count')

plt.show()
# Convert Date column to datetime

train.Date = train.Date.astype('datetime64[ns]')

test.Date = test.Date.astype('datetime64[ns]')
# The store should be open in the test, so we fillna with 1

test.Open.fillna(1, inplace=True)
# Replace missing values with the mode

train['CompetitionDistance'] = train['CompetitionDistance'].fillna(train['CompetitionDistance'].mode()[0])



#Double check if we still see nulls for the column

train['CompetitionDistance'].isnull().sum()/train.shape[0] * 100
# Replace missing values with the mode

test['CompetitionDistance'] = test['CompetitionDistance'].fillna(test['CompetitionDistance'].mode()[0])



#Double check if we still see nulls for the column

test['CompetitionDistance'].isnull().sum()/test.shape[0] * 100
# Extract some properties from Date column

def properties_create(df):

    

    df['Year'] = df.Date.dt.year

    df['Month'] = df.Date.dt.month

    df['Day'] = df.Date.dt.day

    df['Week'] = df.Date.dt.week

    df['Quarter'] = df.Date.dt.quarter

    

    df['Season'] = np.where(df['Month'].isin([3,4,5]), "Spring",

                            np.where(df['Month'].isin([6,7,8]), "Summer",

                                np.where(df['Month'].isin([9,10,11]), "Fall",

                                    np.where(df['Month'].isin([12,1,2]), "Winter", "None"))))

    

    print('Properties creation done!')

    

    return df
train = properties_create(train)
test = properties_create(test)
print(train[['Date', 'Year', 'Month', 'Day', 'Week', 'Quarter', 'Season']].head())
# Create the bar plot for Average Sales across different Seasons

ax = sns.barplot(data=train, x='Season', y='Sales', color='steelblue') 
# Preprocessing before feature creation

train_dim = train.shape[0]



# New dataframe

df_new = train.append(test, ignore_index = True)
# Feature creation



# Define a variable for each type of feature 

target = ['Sales']

numeric_columns = ['Open', 'Promo', 'Promo2', 'StateHoliday', 'SchoolHoliday', 'CompetitionDistance']

categorical_columns = ['DayOfWeek', 'Quarter', 'Month', 'Year', 'StoreType', 'Assortment', 'Season']



# Define a function that will intake the raw dataframe and the column name and return a one hot encoded DF

def create_ohe(df, col):

    le = LabelEncoder()

    a = le.fit_transform(df_new[col]).reshape(-1,1)

    ohe = OneHotEncoder(sparse=False)

    column_names = [col+ "_"+ str(i) for i in le.classes_]

    return(pd.DataFrame(ohe.fit_transform(a), columns=column_names))



# Since the above function coverts the column, one at a time

# We create a loop to create the final dataset with all features

temp = df_new[numeric_columns]

for column in categorical_columns:

    temp_df = create_ohe(df_new, column)

    temp = pd.concat([temp, temp_df], axis=1)



# Add Sales to dataframe

temp_df = df_new[target]

temp = pd.concat([temp, temp_df], axis=1)

df_new = temp
# Fix the object type

print(df_new.columns[df_new.dtypes=='object'])
df_new['StateHoliday'].unique()
df_new['StateHoliday'] = np.where(df_new['StateHoliday']=='0', 0, 1)

# One last check the data type

df_new.dtypes.unique()
# Kaggle: divide dataset of train and test

train = df_new[ : train_dim ]

test = df_new[ train_dim  : ]



# Drop the features of not help

test.drop(['Sales'], axis=1, inplace=True)
# See the column data types after new features

train.info()
# See the column data types after new features

test.info()
# Separate out the features and targets

features = train.drop(columns='Sales')

targets = pd.DataFrame(train['Sales'])



# Split into training and testing set

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)



# Split training set into train and validation set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)



print('Shape of X_train:', X_train.shape)

print('Shape of X_test:', X_test.shape)

print('Shape of X_val:', X_val.shape)

print('Shape of y_train:', y_train.shape)

print('Shape of y_test:', y_test.shape)

print('Shape of y_val:', y_val.shape)
features.columns
targets.columns
# Create the scaler object with a range of 0-1

scaler = MinMaxScaler(feature_range=(0, 1))



# Fit on the training data

scaler.fit(X_train)



# Transform both the training, testing and validation data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

X_val = scaler.transform(X_val)
# Calculate the average score of the train dataset

mean_sales = y_train.mean()

print("Average Sales:", mean_sales) 
# Calculate the Mean Absolute Error on the test dataset

print("MAE for Test Data:", abs(y_test - mean_sales).mean())
from keras.callbacks import History 

history = History()



model = Sequential()

model.add(Dense(350, input_dim = 43, activation="relu"))

model.add(Dense(350, activation="relu"))

model.add(Dense(350, activation="relu"))

model.add(Dense(350, activation="relu"))

model.add(Dense(350, activation="relu"))

model.add(Dense(1, activation="linear"))



model.compile(optimizer='adam', loss="mean_squared_error", metrics=["mean_absolute_error"])



model.fit(X_train, y_train, validation_data=(X_val, y_val), 

epochs=100, batch_size=64, callbacks=[history])



result = model.evaluate(X_test, y_test)



for i in range(len(model.metrics_names)):

    print("Metric ", model.metrics_names[i], ":", str(round(result[i],2)))
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model's Training & Validation loss across epochs")

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()
# Manually predicting from the model, instead of using model's evaluate function

y_test["Prediction"] = model.predict(X_test)

y_test.columns = ["Actual Sales", "Predicted Sales"]

print(y_test.head(10))



# Manually predicting from the model, instead of using model's evaluate function

from sklearn.metrics import mean_squared_error, mean_absolute_error

print("MSE :", mean_squared_error(y_test["Actual Sales"].values, y_test["Predicted Sales"].values))

print("MAE :", mean_absolute_error(y_test["Actual Sales"].values, y_test["Predicted Sales"].values))
# Read in data into a dataframe 

submission = pd.read_csv('../input/sample_submission.csv', sep=',')
# Id submission

index = submission['Id']
# Collecting submission dataset

X_submission = test
X_submission.info()
X_submission.head()
# Create the scaler object with a range of 0-1

scaler = MinMaxScaler(feature_range=(0, 1))



# Fit on the submission data

scaler.fit(X_submission)



# Transform the submission data

X_submission = scaler.transform(X_submission)
# Make predictions on the submission set

model_pred = model.predict(X_submission)

model_pred
# Final dataset with predictions

submission = pd.DataFrame()

submission['Id'] = index

submission['Sales'] = model_pred



submission.to_csv('submission.csv', index = False)
submission.head(20)