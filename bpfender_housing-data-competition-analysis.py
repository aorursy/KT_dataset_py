import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Load test data

train_fp = "../input/train.csv"

train_data = pd.read_csv(train_fp)



test_fp = "../input/test.csv"

test_data = pd.read_csv(test_fp)



y_full = train_data.SalePrice

X_full = train_data.drop(["SalePrice", "Id"],axis=1)



# Seperate categorical and numberical data. Some categorical data uses numbers and isn't identified here e.g MSSubClass,

# OverallQual, OverallCond etc. These are dealt with on a case by case basis.

numerical_cols = [col for col in X_full.columns if X_full.dtypes[col] != 'object']

text_cols = [col for col in X_full.columns if X_full.dtypes[col] == 'object']



# As discussed above, MSSubClass is definitely categorical. Quality ratings don't disturb me for the moment as they are

# quasi-numberical and have clear ranking/order

numerical_cols.remove('MSSubClass')

text_cols.insert(0,'MSSubClass')



print("Numerical column list:\n", numerical_cols, "\n")

print("Text column list:\n",text_cols, "\n")
#Create correlation heatmap of numberical values

cor_list = numerical_cols.copy()

cor_list.insert(0, 'SalePrice')



num_corr = train_data[cor_list].corr(method='spearman')

fig,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data=num_corr, square=True)
#Sort correlation with sale price in order and display

num_corr.sort_values(['SalePrice'], ascending=False, inplace=True)

print(num_corr.SalePrice)
#Display more detailed version looking at top-n variable

k = 20

cols = num_corr.nlargest(k,'SalePrice').index

num_corr_top = train_data[cols].corr(method='spearman')

fig,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data=num_corr_top, square=True, annot=True)

#Plot scatter plots of numerical data vs price to look at general trends

f= pd.melt(train_data,id_vars=['SalePrice'], value_vars=numerical_cols)

g = sns.FacetGrid(f, col='variable', col_wrap=2, sharey=False, sharex=False, height = 4, aspect = 1.5)

g = g.map(sns.scatterplot, "value", "SalePrice")
# Examing a few different suggestions from above

# Combining basement area and living area for a total area 

from scipy.stats import spearmanr

total_area = train_data['GrLivArea']+train_data['TotalBsmtSF']

total_area2 = total_area + train_data['GarageArea']



corr,p_value = spearmanr(train_data['GrLivArea'],y_full)

print('Total Area (GrLivArea) Spearman:',corr)

corr,p_value = spearmanr(total_area,y_full)

print('Total Area (GrLivArea and TotalBsmtSF) Spearman:',corr)

corr,p_value = spearmanr(total_area2,y_full)

print('Total Area (GrLivArea, TotalBsmtSF and GarageArea:',corr)



sns.scatterplot(x=total_area2,y=y_full)
# Given that a house is unlikely to have all porch types, it may just make sense to look at them all together.

porch_area = train_data[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)

corr,p_value = spearmanr(porch_area,y_full)

print('Total Porch Area Spearman:',corr)



sns.scatterplot(x=porch_area,y=y_full)
total_baths = train_data[['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']].sum(axis=1)

corr,p_value = spearmanr(total_baths,y_full)

print('Total Baths Spearman:',corr)



sns.scatterplot(x=total_baths,y=y_full)
# NEXT: CATEGORICAL DATA

# Plot scatter plots of numerical data vs price to look at general trends

f= pd.melt(train_data.fillna('MISSING'),id_vars=['SalePrice'], value_vars=text_cols)

g = sns.FacetGrid(f, col='variable', col_wrap=2, sharey=False, sharex=False, height = 4, aspect = 1.5)

g = g.map(sns.boxplot, "value", "SalePrice")



#for i in range(len(text_cols)):

 #   var = text_cols[i]

  #  plt.title(var)

   # #Fills in NA with "MISSING" placeholder so they are displayed on the graph

    #sns.boxplot(x=X_full[var].fillna('MISSING'),y=y_full)

    #plt.show()
# Create a copy of the training data to start processing it.

train_data_copy = train_data.copy()
nbrhd_means = train_data_copy['SalePrice'].groupby(train_data_copy['Neighborhood']).mean().sort_values()

nbrhd_index = nbrhd_means.index.values

#print(nbrhd_index)



for i in range(len(nbrhd_index)):

    train_data_copy = train_data_copy.replace({'Neighborhood':{nbrhd_index[i]:i}})



sns.scatterplot(train_data_copy['Neighborhood'],train_data_copy['SalePrice'])

corr,p_value = spearmanr(train_data_copy['Neighborhood'],train_data_copy['SalePrice'])

print('Neighborhood Ranked Spearman:',corr)
style_means = train_data_copy['SalePrice'].groupby(train_data_copy['HouseStyle']).mean().sort_values()

style_index = style_means.index.values



for i in range(len(style_index)):

     train_data_copy = train_data_copy.replace({'HouseStyle':{style_index[i]:i}})



sns.scatterplot(train_data_copy['HouseStyle'],train_data_copy['SalePrice'])

corr,p_value = spearmanr(train_data_copy['HouseStyle'],y_full)

print('House Style Ranked Spearman:',corr)
# Need to handle filling in missing values before doing this stuff

# Originally from https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset, with some edits

# Have removed some categories where I don't feel that they necesarrilly correspond to a measure of quality, and in turn can't

# really be ranked



# For these ranked categories, fill in the missing values first so that rankings are applied to the complete data set. Various

# assumptions made about what missing values mean

train_data_copy = train_data_copy.fillna({

    "Alley" : "No", #NAs defined as not present

    "BsmtCond" : "No",

    "BsmtExposure" : "No",

    "BsmtFinType1" :"No",

    "BsmtFinType2" :"No",

    "BsmtQual" : "No",

    "ExterCond" :"TA", #assume that if no quality measure, quality is average

    "ExterQual" :"TA",

    "FireplaceQu" :"TA",

    "Functional" : "Typ", #assume typical unless dedcutions are warranted

    "GarageCond" : "No", #assume no garage present if NA

    "GarageQual" : "No",

    "GarageFinish" :"No",

    "HeatingQC" : "TA", #assume average condition

    "KitchenQual" : "TA",

    "LotShape" : "Reg", #assume regular

    "PavedDrive" : "Y", #dataset suggests the majority of drives are paved

    "PoolQC" : "No",

    "Street" : "Pave", #assume that most streets are paved (this plays out in the dataset)

    "Utilities" : "AllPub", #dataset suggests that majority of properties have all utilities 

    "CentralAir" : "N",

    "Electrical" : "SBrkr",

    "LandSlope" : "Gtl"

})



# Convert text based rankings to numerical ones

train_data_copy = train_data_copy.replace({

    "Alley" : {"No": 0, "Grvl" : 1, "Pave" : 2},

    "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

    "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,"ALQ" : 5, "GLQ" : 6},

    "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,"ALQ" : 5, "GLQ" : 6},

    "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

    "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},

    "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageFinish" : {"No" : 0, "Unf":1, "RFn":2, "Fin":3},

    "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "LotShape" : {"IR3" : 4, "IR2" : 3, "IR1" : 2, "Reg" : 1},

    "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

    "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

    "Street" : {"Grvl" : 1, "Pave" : 2},

    "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},

    "CentralAir" : {"N" : 0, "Y":1},

    "Electrical" : {"Mix": 1, "FuseP":2, "FuseF":3, "FuseA":4, "SBrkr":5},

    "LandSlope" :{"Sev":1, "Mod":2, "Gtl":3}

})
#Check for columns where data is missing after 'valid' NAs are processed above

missing = train_data_copy.isnull().sum()

missing = missing[missing>0]

missing.sort_values(ascending=False, inplace=True)



print(len(missing),"columns have missing values:\n")

print(missing,"\n")
# Drop non-reqired columns now to avoid generating excessive entries

drop_list = ['Id','MSSubClass', 

             'Heating', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Condition2', 'MiscFeature',

             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'#cheeky hardcoding to remove features that aren't present in final test datset

             #'Fence', 'GarageYrBlt', #dropping missing columns that definitely won't be used

             #'YrSold', 'BsmtFinSF2', 'MiscVal', 'LowQualFinSF', 'OverallCond',

             #'KitchenAbvGr' ,#remove low correlation numerical vars (-ve)

             #'PoolArea', 'MoSold', 'BsmtUnfSF', 'PoolQC', 'BsmtCond', 'PavedDrive',#remove low correlation numerical vars (+ve)

             #'ExterCond','Utilities', 'Street', 'BsmtFinType2','Functional', 'Electrical','FireplaceQu',  #remove low correlations cats(+ve)

             #'LandSlope','Alley',# remove low correlation cat(-ve)

             #'LotConfig', 'SaleCondition', 'SaleType','RoofStyle', 'Condition1', 'MSZoning',

             #'LandContour', 'BldgType', 'Foundation', 'GarageType', 'MasVnrType',#remove 1H encoded variables, before they are encoded

             #'BsmtFinSF1','LotFrontage', '1stFlrSF', '2ndFlrSF', 'GarageCond', 'BedroomAbvGr',

             #'BsmtQual', 'ExterQual', # remove variable that seem to correlate highly within dataset

             #

            ]



train_data_final = train_data_copy.drop(drop_list, axis=1)



train_data_final['TotalArea'] = train_data_copy[['GrLivArea','TotalBsmtSF', 'GarageArea']].sum(axis=1)

train_data_final['TotalBaths'] = train_data_copy[['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']].sum(axis=1)



train_data_final['OverallQual'] = train_data_final['OverallQual'] ** 2 

train_data_final['Neighborhood'] = train_data_final['Neighborhood'] ** 2 #weighting these more highly as they appear to follow a quadratic relationship



# Drop data that has been used above to calculate total quantities

train_data_final = train_data_final.drop(['GrLivArea','TotalBsmtSF', 'GarageArea',

                                        'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'],axis=1)



# One hot encode data

OH_train_data_final = pd.get_dummies(train_data_final)



# Remove outliers identified earlier

OH_train_data_final = OH_train_data_final[OH_train_data_final.TotalArea < 8000]
#Look at correlations in shortened dataset with modified category values

correlated_complete = OH_train_data_final.corr(method='spearman')



correlated_complete.sort_values(['SalePrice'], ascending=False, inplace=True)

print(correlated_complete.SalePrice)



k = 50

cols_complete = correlated_complete.nlargest(k,'SalePrice').index

correlated_complete_top = OH_train_data_final[cols_complete].corr(method='spearman')



#print(correlated_complete_top.SalePrice)

fig,ax = plt.subplots(figsize=(16,16))

sns.heatmap(data=correlated_complete_top, square=True, annot=True)
import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

import math

import numpy as np

from sklearn.metrics import mean_absolute_error



X_full = OH_train_data_final.drop(['SalePrice'],axis=1)

y_full = OH_train_data_final['SalePrice']



# Run basic imputation to fill in missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()



X_full_imp = pd.DataFrame(imputer.fit_transform(X_full))

X_full_imp.columns = X_full.columns



dtrain = xgb.DMatrix(X_full_imp, label = np.log(y_full))

params = {"max_depth":2, "eta":0.1} #parameters tuned by trial and error, would like to have a more systematic approach

model = xgb.cv(params, dtrain,  num_boost_round=2000, early_stopping_rounds=50, metrics=['rmse'])

model.loc[50:,["test-rmse-mean", "train-rmse-mean"]].plot()

model.tail()
X_train, X_valid, y_train, y_valid = train_test_split(X_full,y_full, train_size=0.8, test_size = 0.2, random_state=0)



X_train_imp = pd.DataFrame(imputer.fit_transform(X_train))

X_valid_imp = pd.DataFrame(imputer.transform(X_valid))

X_train_imp.columns = X_train.columns

X_valid_imp.columns = X_valid.columns



my_model = XGBRegressor(max_depth = 2, n_estimators=2000, learning_rate=0.1, random_state=0)

my_model.fit(X_train, np.log(y_train),early_stopping_rounds= 50, eval_set=[(X_valid,np.log(y_valid))], verbose=False)



predictions = np.exp(my_model.predict(X_valid))

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
#Train model on full training dataset

my_model_final = XGBRegressor(max_depth = 2, n_estimators = 355, learning_rate = 0.1) #params taken from XGBcv

my_model_final.fit(X_full_imp, np.log(y_full))
#THIS IS NASTY AND SHOULD VERY MUCH BE CONTAINED IN FUNCTIONS, APOLOGIES

#Preprocessing of test_data

test_data_proc = test_data.copy()

for i in range(len(nbrhd_index)):

    test_data_proc = test_data_proc.replace({'Neighborhood':{nbrhd_index[i]:i}})



for i in range(len(style_index)-1): #slightly cheeky hard code as last index isn't in test set

    test_data_proc = test_data_proc.replace({'HouseStyle':{style_index[i]:i}})



test_data_proc = test_data_proc.fillna({

    "Alley" : "No", #NAs defined as not present

    "BsmtCond" : "No",

    "BsmtExposure" : "No",

    "BsmtFinType1" :"No",

    "BsmtFinType2" :"No",

    "BsmtQual" : "No",

    "ExterCond" :"TA", #assume that if no quality measure, quality is average

    "ExterQual" :"TA",

    "FireplaceQu" :"TA",

    "Functional" : "Typ", #assume typical unless dedcutions are warranted

    "GarageCond" : "No", #assume no garage present if NA

    "GarageQual" : "No",

    "GarageFinish" :"No",

    "HeatingQC" : "TA", #assume average condition

    "KitchenQual" : "TA",

    "LotShape" : "Reg", #assume regular

    "PavedDrive" : "Y", #dataset suggests the majority of drives are paved

    "PoolQC" : "No",

    "Street" : "Pave", #assume that most streets are paved (this plays out in the dataset)

    "Utilities" : "AllPub", #dataset suggests that majority of properties have all utilities 

    "CentralAir" : "N",

    "Electrical" : "SBrkr",

    "LandSlope" : "Gtl"

})



test_data_proc = test_data_proc.replace({

    "Alley" : {"No": 0, "Grvl" : 1, "Pave" : 2},

    "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

    "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,"ALQ" : 5, "GLQ" : 6},

    "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,"ALQ" : 5, "GLQ" : 6},

    "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

    "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

    "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},

    "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "GarageFinish" : {"No" : 0, "Unf":1, "RFn":2, "Fin":3},

    "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

    "LotShape" : {"IR3" : 4, "IR2" : 3, "IR1" : 2, "Reg" : 1},

    "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

    "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

    "Street" : {"Grvl" : 1, "Pave" : 2},

    "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},

    "CentralAir" : {"N" : 0, "Y":1},

    "Electrical" : {"Mix": 1, "FuseP":2, "FuseF":3, "FuseA":4, "SBrkr":5},

    "LandSlope" :{"Sev":1, "Mod":2, "Gtl":3}

})



test_data_proc = test_data_proc.drop(drop_list, axis=1)



test_data_proc['TotalArea'] = test_data_proc[['GrLivArea','TotalBsmtSF', 'GarageArea']].sum(axis=1)

test_data_proc['TotalBaths'] = test_data_proc[['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']].sum(axis=1)



train_data_final['OverallQual'] = train_data_final['OverallQual'] ** 2 

train_data_final['Neighborhood'] = train_data_final['Neighborhood'] ** 2 



test_data_proc = test_data_proc.drop(['GrLivArea','TotalBsmtSF', 'GarageArea',

                                        'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'],axis=1)



test_data_proc = pd.get_dummies(test_data_proc)

test_data_proc.index = test_data["Id"]
#Impute any missing values

test_data_proc_imp = pd.DataFrame(imputer.transform(test_data_proc))

test_data_proc_imp.columns = test_data_proc.columns


test_data_proc_imp.index = test_data["Id"]



preds_test = np.exp(my_model_final.predict(test_data_proc_imp))



# Save test predictions to file

output = pd.DataFrame({'Id': test_data_proc_imp.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)