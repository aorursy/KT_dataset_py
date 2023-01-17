# importing two basic libraries

import pandas as pd

import os
# Print list of available files

print(os.listdir("../input"))
# Import train.csv file in df_train dataframe

df_train = pd.read_csv("../input/train.csv")



# Print all the columns available from file:

df_train.columns
# How to calculate total missing values

total = df_train.isnull().sum().sort_values(ascending=False)



# Calculate percentage of missing values

percent = ((df_train.isnull().sum()/df_train.isnull().count())*100).sort_values(ascending=False)



# Presenting in tabular form by concatinating both values and create a seperate data set

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])



# Now check top results of missing_data

missing_data.head(20)
# Copying data frame

df_train_delrows = df_train
# All rows with less than 5% and greater than 0% of missing data

missing_data[(missing_data['Percent']<5) & (missing_data['Percent']>0)]
# We can select rows by using below statement and then use it seperatly for each feature rows:

df_train_delrows[df_train_delrows['Electrical'].isnull()]
# Deleting rows based on our above statement:

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtExposure'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtFinType2'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtFinType1'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtCond'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['BsmtQual'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['MasVnrArea'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['MasVnrType'].isnull()]).index,0)

df_train_delrows = df_train_delrows.drop((df_train_delrows[df_train_delrows['Electrical'].isnull()]).index,0)
# Now verify if there is any missing data left for any of these columns

df_train_delrows['BsmtQual'].isnull().sum()

df_train_delrows['MasVnrArea'].isnull().sum()
# Let's see missing_data dataset

missing_data.head(20)
# Loading a seperate dataset for mean

df_train_men = df_train
# Calculate Mean

df_train_men['MasVnrArea'].mean()
df_train_men[df_train_men['MasVnrArea'].isnull()]
# now we can fill NaN records with Mean values

df_train_men['MasVnrArea'].fillna(df_train_men['MasVnrArea'].mean(), inplace=True)
# Verify if NaN records still exists or not

df_train_men[df_train_men['MasVnrArea'].isnull()]
# Load df_train in seperate dataframe

df_train_null = df_train
df_train_null[df_train_null['MasVnrType'].isnull()]
# Percentage of missing values:

missing_data.head(20)
# Dropping entire feature

df_train_null = df_train_null.drop((missing_data[missing_data['Percent']>15]).index,axis=1)
# Verify if columns exist or not

df_train_null.isnull().sum().sort_values(ascending=False).head(20)