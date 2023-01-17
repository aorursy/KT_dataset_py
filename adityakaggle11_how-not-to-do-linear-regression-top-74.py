# Import all the required libraries

import pandas  as pd #Data manipulation

import numpy as np #Data manipulation

import matplotlib.pyplot as plt # Visualization

import seaborn as sns #Visualization

plt.rcParams['figure.figsize'] = [8,5]

plt.rcParams['font.size'] =14
# Import the data

import os;



# Print the contents of the current directory

print(os.listdir());



trainDatasetName = '../input/train.csv';



print("Reading data from: " + trainDatasetName);

df = pd.read_csv(trainDatasetName);

print("Done reading data from " + trainDatasetName);



testDatasetName = '../input/test.csv';



print("Reading data from: " + testDatasetName);

df_test = pd.read_csv(testDatasetName);

print("Done reading data from " + testDatasetName);
# Printing the first 5 rows from the dataset

df.head(10)
# Printing the shape and assigning the rows and columns in the dataset

nRow, nCol = df.shape;

nRow_test, nCol_test = df_test.shape;

print(df.shape);

print(df_test.shape)
# Print the columns in the dataset

df.columns
# Get a description of the whole dataset across all columns here

# The thing to notice is you can find if some column is using 0 as a placeholder

# for the NaN/ Missing or Null values;

df.describe()
# Get Information about the dataset, 

# most important thing to notice here is which columns are of type 

# object as they will be require to encoded

"""

    There are many independent variables of type object

"""

df.info()
# Plot a correlation plot

corr = df.corr();

plt.figure(figsize=(20,10))

sns.heatmap(corr, cmap='Wistia', annot=True);
print("Let's start by plotting a regression line between two axis in dataset..");



"""

    From the correlation plot we noticed that the Top correlated independent columns with our SalePrice

    are:

"""



fig = plt.figure(figsize=(20,10));



# We will try 'regplot' by seaborn



ax = fig.add_subplot(231);

sns.regplot(x=df.OverallQual, y=df.SalePrice, data=df, marker='*', color='purple');

plt.title('OverallQual vs SalePrice');



ax = fig.add_subplot(232);

sns.regplot(x=df.TotalBsmtSF, y=df.SalePrice, data=df, marker='*', color='purple');

plt.title('TotalBsmtSF vs SalePrice');



ax = fig.add_subplot(233);

sns.regplot(x='1stFlrSF', y=df.SalePrice, data=df, marker='*', color='purple');

plt.title('1stFlrSF vs SalePrice');



ax = fig.add_subplot(234);

sns.regplot(x=df.GrLivArea, y=df.SalePrice, data=df, marker='*', color='purple');

plt.title('GrLivArea vs SalePrice');



ax = fig.add_subplot(235);

sns.regplot(x=df.GarageCars, y=df.SalePrice, data=df, marker='*', color='purple');

plt.title('GarageCars vs SalePrice');



ax = fig.add_subplot(236);

sns.regplot(x=df.GarageArea, y=df.SalePrice, data=df, marker='*', color='purple');

plt.title('GarageArea vs SalePrice');

#----Data Preprocessing----#

# Find the percentage of nulls in each of the columns

x = df.isnull().sum() / nRow * 100

x.sort_values(ascending=False)
# Check for missing value and plot it over a heatmap

plt.figure(figsize=(20,10))

sns.heatmap(df.isnull(),cbar=False,cmap='viridis',yticklabels=False)

plt.title('Missing value in the dataset')
#----Data Processing----#

print(df.LotFrontage.head(10));

print("Percentage of Nulls: ", df.LotFrontage.isnull().sum() / nRow * 100);

sns.distplot(df.LotFrontage.dropna())

"""

    Here you can see that the maximum count of LotFrontage column lies in between 50-100

    Hence best if can find some stastic number to impudiate the NaN value which lies between 50-100

"""



print("Mean of LotFrontage: ", df.LotFrontage.mean());

print("Median of LotFrontage: ", df.LotFrontage.median());



df.LotFrontage.fillna(df.LotFrontage.mean(), inplace=True);



print(df.LotFrontage.head(10))
#----Data Processing----#

print(df.Alley.head(10));

print("Percentage of Nulls: ", df.Alley.isnull().sum() / nRow * 100);



"""

    Since percentage of NaN is > 50% hence best if we could drop this column

"""

print("Dropping alley: ", df.shape[1]);

df = df.drop('Alley', axis=1)

print("After dropping: ", df.shape[1]);
#----Data Processing----#

print(df.MasVnrType.head(10));

print("Percentage of Nulls: ", df.MasVnrType.isnull().sum() / nRow * 100);



"""

    Since this column is a categorical value hence we might need to impute the NaN's with the value

    which has the maximum count of values

"""



print(df.MasVnrType.value_counts());



df.MasVnrType.fillna('BrkFace', inplace=True);



print("Percentage of Nulls: ", df.MasVnrType.isnull().sum() / nRow * 100);
#----Data Processing----#

print(df.MasVnrArea.head(10));

print("Percentage of Nulls: ", df.MasVnrArea.isnull().sum() / nRow * 100);

sns.distplot(df.MasVnrArea.dropna())

"""

    Here you can see that the maximum count of MasVnrArea column lies in between 50-100

    Hence best if can find some stastic number to impudiate the NaN value which lies between 50-100

"""



print("Mean of LotFrontage: ", df.MasVnrArea.mean());

print("Median of LotFrontage: ", df.MasVnrArea.median());



df.MasVnrArea.fillna(df.MasVnrArea.mean(), inplace=True);



print("Percentage of Nulls: ", df.MasVnrArea.isnull().sum() / nRow * 100);
"""

    Going fastract with some columns

"""

df.BsmtQual.fillna('TA', inplace=True);



df.BsmtCond.fillna('TA', inplace=True);



df.BsmtExposure.fillna('No', inplace=True);



df.BsmtFinType1.fillna('Unf', inplace=True);



df.BsmtFinType2.fillna('Unf', inplace=True);



df.Electrical.fillna('SBrkr', inplace=True);



df.FireplaceQu.fillna('Gd', inplace=True);



df.GarageType.fillna('Attchd', inplace=True);



df.GarageYrBlt.fillna(df.GarageYrBlt.mean(), inplace=True);



df.GarageFinish.fillna('Unf', inplace=True);



df.GarageQual.fillna('TA', inplace=True);



df.GarageCond.fillna('TA', inplace=True);



df = df.drop('PoolQC', axis=1);



df = df.drop('Fence', axis=1);



df = df.drop('MiscFeature', axis=1);
# Let's check the percentage of nulls after imputing and removal

#----Data Preprocessing----#

# Find the percentage of nulls in each of the columns

x = df.isnull().sum() / nRow * 100

x.sort_values(ascending=False)
# To perform same data processing and cleansing for df_test

print(df_test.shape)

print(df_test.isnull().sum() / df_test.shape[0] * 100)
# Check for missing value and plot it over a heatmap

plt.figure(figsize=(20,10))

sns.heatmap(df_test.isnull(),cbar=False,cmap='viridis',yticklabels=False)

plt.title('Missing value in the dataset')
#----Data Processing----#

print(df_test.LotFrontage.head(10));

print("Percentage of Nulls: ", df_test.LotFrontage.isnull().sum() / nRow * 100);

sns.distplot(df_test.LotFrontage.dropna())

"""

    Here you can see that the maximum count of LotFrontage column lies in between 50-80

    Hence best if can find some stastic number to impudiate the NaN value which lies between 50-100

"""



print("Mean of LotFrontage: ", df_test.LotFrontage.mean());

print("Median of LotFrontage: ", df_test.LotFrontage.median());



df_test.LotFrontage.fillna(df_test.LotFrontage.mean(), inplace=True);



print(df_test.LotFrontage.head(10))
#----Data Processing----#

print(df_test.Alley.head(10));

print("Percentage of Nulls: ", df_test.Alley.isnull().sum() / nRow * 100);



"""

    Since percentage of NaN is > 50% hence best if we could drop this column

"""

print("Dropping alley: ", df_test.shape[1]);

df_test = df_test.drop('Alley', axis=1)

print("After dropping: ", df_test.shape[1]);
#----Data Processing----#

print(df_test.MasVnrType.head(10));

print("Percentage of Nulls: ", df_test.MasVnrType.isnull().sum() / nRow * 100);



"""

    Since this column is a categorical value hence we might need to impute the NaN's with the value

    which has the maximum count of values

"""



print(df_test.MasVnrType.value_counts());



df_test.MasVnrType.fillna('BrkFace', inplace=True);



print("Percentage of Nulls: ", df_test.MasVnrType.isnull().sum() / nRow * 100);
#----Data Processing----#

print(df_test.MasVnrArea.head(10));

print("Percentage of Nulls: ", df_test.MasVnrArea.isnull().sum() / nRow * 100);

sns.distplot(df_test.MasVnrArea.dropna())

"""

    Here you can see that the maximum count of MasVnrArea column lies in between 50-100

    Hence best if can find some stastic number to impudiate the NaN value which lies between 50-100

"""



print("Mean of MasVnrArea: ", df_test.MasVnrArea.mean());

print("Median of MasVnrArea: ", df_test.MasVnrArea.median());



df_test.MasVnrArea.fillna(df_test.MasVnrArea.mean(), inplace=True);



print("Percentage of Nulls: ", df_test.MasVnrArea.isnull().sum() / nRow * 100);
"""

    Going fastract with some columns

"""

df_test.BsmtQual.fillna('TA', inplace=True);

df_test.BsmtCond.fillna('TA', inplace=True);

df_test.BsmtExposure.fillna('No', inplace=True);

df_test.BsmtFinType1.fillna('Unf', inplace=True);

df_test.BsmtFinType2.fillna('Unf', inplace=True);

df_test.FireplaceQu.fillna('Gd', inplace=True);

df_test.GarageType.fillna('Attchd', inplace=True);

df_test.GarageYrBlt.fillna(df.GarageYrBlt.mean(), inplace=True);

df_test.GarageFinish.fillna('Unf', inplace=True);

df_test.GarageQual.fillna('TA', inplace=True);

df_test.GarageCond.fillna('TA', inplace=True);

df_test = df_test.drop('PoolQC', axis=1);

df_test = df_test.drop('Fence', axis=1);

df_test = df_test.drop('MiscFeature', axis=1);



df_test.MSZoning.fillna('RL', inplace=True);

df_test.Utilities.fillna('AllPub', inplace=True);

df_test.Exterior1st.fillna('VinylSd', inplace=True);

df_test.Exterior2nd.fillna('VinylSd', inplace=True);

df_test.BsmtFinSF1.fillna(df.BsmtFinSF1.mean(), inplace=True);

df_test.BsmtFinSF2.fillna(df.BsmtFinSF2.mean(), inplace=True);

df_test.BsmtUnfSF.fillna(df.BsmtUnfSF.mean(), inplace=True);

df_test.TotalBsmtSF.fillna(df.TotalBsmtSF.mean(), inplace=True);

df_test.BsmtFullBath.fillna(df.BsmtFullBath.mean(), inplace=True);

df_test.BsmtHalfBath.fillna(df.BsmtHalfBath.mean(), inplace=True);

df_test.KitchenQual.fillna('TA', inplace=True);

df_test.Functional.fillna('Typ', inplace=True);

df_test.GarageCars.fillna(df.GarageCars.mean(), inplace=True);

df_test.GarageArea.fillna(df.GarageArea.mean(), inplace=True);

df_test.SaleType.fillna('WD', inplace=True);
df_test.isnull().sum() / df_test.shape[0] * 100
X_train = df.drop('SalePrice', axis=1)

X_train = X_train.select_dtypes(exclude=['object'])

y_train = df.SalePrice
X_test = df_test.select_dtypes(exclude=['object'])



from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()

model.fit(X_train, y_train)

preds = model.predict(X_test)



submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': preds})

submission.to_csv('submission.csv', index=False)