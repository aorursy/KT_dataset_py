# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt   #Data visualisation libraries 

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Create a dataframe with the entire dataset; we will later split this data later during the regression step

house_data = pd.read_csv("../input/train.csv")

house_data.head()
#Create dataframe with numeric features

house_data_numeric_fields = house_data[['LotFrontage','LotArea','OverallQual','OverallCond',

                            'YearBuilt','YearRemodAdd','MasVnrArea',

                             'YrSold','MoSold','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','TotRmsAbvGrd',

                             'BedroomAbvGr','FullBath','Fireplaces',

                             'GarageYrBlt','GarageCars','GarageArea','GrLivArea','SalePrice']]



#Display sample rows from above dataframe

house_data_numeric_fields.head()
#Create correlation dataframe

corr = house_data_numeric_fields.corr()



#Configure figure size

fig, ax = plt.subplots(figsize=(10, 10))



#Produce colour map

colormap = sns.diverging_palette(220, 10, as_cmap=True)



#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")



#Configure x ticks

plt.xticks(range(len(corr.columns)), corr.columns);



#Configure y ticks

plt.yticks(range(len(corr.columns)), corr.columns)



#show plot

plt.show()


# Create a dataframe with numeric attributes with most correlation with the sale price

house_data_numeric_high_corr = house_data_numeric_fields[['GrLivArea', 'GarageArea','GarageCars','GarageYrBlt','FullBath',

             'TotRmsAbvGrd', 'TotalBsmtSF','YearRemodAdd','YearBuilt','OverallQual','Fireplaces','SalePrice']]
# Plot a scatter matrix



pd.plotting.scatter_matrix(house_data_numeric_high_corr,alpha=0.2, figsize=(20, 20))

plt.show()
# identify all columsn which  have null values

house_data_numeric_high_corr_nulls = house_data_numeric_high_corr.loc[:, house_data_numeric_high_corr.isna().any()]



# The year when the garage was built had nulls; it was also observed that the mode of the feature was year was 2005, hence the 

# nulls were replaced by the statistical mode, i.e.year 2005

df_temp = house_data_numeric_high_corr.fillna({'GarageYrBlt':2005})



# All the other features with nulls, except for the garage year built, which was fixed in the above step, were dropped

house_data_numeric_high_corr_cleaned = df_temp.dropna(axis=0,how='any')



# Display sample rows from the cleaned up dataset with no null values

house_data_numeric_high_corr_cleaned.head()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



# X_true : captures the features in the data set, to which LR will be applied, i.e this the training set

# y_true : captures the sale price in the data set, to which LR will be applied, i.e this the training set



# X_predict : captures the features in the data set, which can be used to verify the model generated using X_true

# y_predict : captures the sale price in the data set, which can be used to verify the model using y_true



linreg = LinearRegression()



# Divide the data into a 30% to 70% split, and use one to train the model, and use the other to predict; hence, training dataset 

# has the sale price; however, sale price has been removed from the dataset to be used for prediction



X_true, X_predict, y_true, y_predict = train_test_split(

                                house_data_numeric_high_corr_cleaned.loc[:, house_data_numeric_high_corr_cleaned.columns != 'SalePrice'],

                                house_data_numeric_high_corr_cleaned['SalePrice'],

                                train_size=.30, test_size=.70)



# Perform curve fitting



linreg = LinearRegression().fit(X_true, y_true)

print('Curve Fitting Complete!')



# Output relevant results from the above step



print('House Prices Dataset')

print('linear model intercept: {}'

     .format(linreg.intercept_))

print('linear model coeff:\n{}'

     .format(linreg.coef_))

print('R-squared score (training): {:.3f}'

     .format(linreg.score(X_true, y_true)))

print('R-squared score (test): {:.3f}'

     .format(linreg.score(X_predict, y_predict)))

predictions = linreg.predict(X_predict)

print('Prediction using the test set based on the curve fitting of the training set is complete!')
plt.scatter(y_predict,predictions)
