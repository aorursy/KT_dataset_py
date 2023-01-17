#conda install py-xgboost

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns

import numpy as np 

import pandas as pd 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import Imputer

from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model

from sklearn.base import TransformerMixin

from sklearn.base import BaseEstimator

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from scipy.stats import skew, skewtest

from subprocess import check_output

from  sklearn.linear_model import LassoCV

from sklearn.base import BaseEstimator, RegressorMixin

from xgboost.sklearn import XGBRegressor



#import the datasets

train = pd.read_csv("../input/train.csv", index_col=0)

df_test = pd.read_csv("../input/test.csv", index_col=0)

target_var = train['SalePrice']  #target variable



#drop the target variable from the training dataset 

df_train = train.drop('SalePrice', axis=1)



#create new coulumn to distinguish the training and test sets for when we split them again

df_train['training_set'] = True

df_test['training_set'] = False



#concatenate the testing and training dataset for data preprocessig 

df_full = pd.concat([df_train, df_test])

df_full.head()

#descriptive statistics summary

target_var.describe()
# Heatmap which shows correlation of features with eachother and the target variable.  

corr = df_full.corr()

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=.8, square=True)

plt.title('Correlation of the features and the target variable');



# Isolate the categorical features from the concatenated dataset.

categoricals = df_full.select_dtypes(include=['object'])



# Determine the number of missing values within each category.

total = categoricals.isnull().sum().sort_values(ascending=False)

# The percentage of missing values. 

percent = round((categoricals.isnull().sum()/categoricals.isnull().count()).sort_values(ascending=False)*100,3)

# Create dataframe of missing value counts as well as their percentage of the whole. 

missing_categorical = pd.concat([total, percent], axis=1, keys=['Total Nulls', 'Percentage (%)'])

missing_categorical.head(30)
# Create a list of categorical features in order to label each bar on the chart. 

labels=list(categoricals.apply(pd.Series.nunique).index)



# Plot the percentage of missing values for each categorical feature. 

percentge=categoricals.applymap(lambda x: pd.isnull(x)).sum()*100/categoricals.shape[0]

plt.figure(figsize=(15,5))

plt.bar(range(len(percentge)), percentge, align='center')

plt.xticks(range(len(labels)), labels, rotation=90)

plt.ylabel("Percentage of null values (%)")

plt.ylim([0,100])

plt.xlim([-1, categoricals.shape[1]])

plt.title("Percentage of nulls for each object feature (%)")

plt.show()
# Plot bar plot for each feature to show the unique categories within each feature and their resepective frequencies.



for category in df_full.dtypes[df_full.dtypes == 'object'].index:

    sns.countplot(y=category, data=df_full)

    plt.title('Frequency of categories within ' + category)

    plt.show()
# Convert numeric categories from integers to strings.

df_full['MSSubClass'] = df_full['MSSubClass'].apply(str)

df_full['YrSold'] = df_full['YrSold'].astype(str)

df_full['MoSold'] = df_full['MoSold'].astype(str)

df_full['YearBuilt'] = df_full['YearBuilt'].astype(str)

df_full['YearRemodAdd'] = df_full['YearRemodAdd'].astype(str) 
# Fill missing values within these features with the most likely category they belong to.  

df_full['Functional'] = df_full['Functional'].fillna('Typ') 

df_full['Electrical'] = df_full['Electrical'].fillna("SBrkr") 

df_full['KitchenQual'] = df_full['KitchenQual'].fillna("TA") 
# Fill these featues missing values with the MODE as it is the most frequent value, and therfore a reasonable assumption. 

df_full['Exterior1st'] = df_full['Exterior1st'].fillna(df_full['Exterior1st'].mode()[0]) 

df_full['Exterior2nd'] = df_full['Exterior2nd'].fillna(df_full['Exterior2nd'].mode()[0])

df_full['SaleType'] = df_full['SaleType'].fillna(df_full['SaleType'].mode()[0])



# Group 'MSZoning' in order to fill nulls with most frequently observed category within their 'MSSubClass'.  

df_full['MSZoning'] = df_full.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# Homes without pools

df_full["PoolQC"] = df_full["PoolQC"].fillna("None")





# Nulls in features relating to a garage are probably due to the lack of one, we fill these values with either 0 or 'None'.  

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df_full[col] = df_full[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    df_full[col] = df_full[col].fillna('None')



    

# Once again, the missing values within basement categories are likely because the house has no basement.  



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_full[col] = df_full[col].fillna('None')
categorical = []

for i in df_full.columns:

    if df_full[i].dtype == object:

        categorical.append(i)

df_full.update(df_full[categorical].fillna('None'))

print(categorical)
# Group homes according to their 'Neighborhood' in order to estimate their 'LotFrontage' with the local median.

df_full['LotFrontage'] = df_full.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# All other missing values within the numeric features are replaced with zero. 

numeric_data_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] # All possible numeric data types to consider.

numerical = []

for i in df_full.columns:

    if df_full[i].dtype in numeric_data_types:

        numerical.append(i)

df_full.update(df_full[numerical].fillna(0))

numerical[1:10]
# We remove any feature that we believe will not influence the prediction of the house price. 

df_full = df_full.drop(['Utilities', 'Street', 'PoolQC','FireplaceQu','MiscFeature'], axis=1)
# Create categorical boolean features based on the presence or absence of particular amenities



df_full['pool?'] = df_full['PoolArea'].apply(lambda x: 'True' if x > 0 else 'False')

df_full['2ndfloor?'] = df_full['2ndFlrSF'].apply(lambda x: 'True' if x > 0 else 'False')

df_full['garage?'] = df_full['GarageArea'].apply(lambda x: 'True' if x > 0 else 'False')

df_full['basement?'] = df_full['TotalBsmtSF'].apply(lambda x: 'True' if x > 0 else 'False')

df_full['fireplace?'] = df_full['Fireplaces'].apply(lambda x: 'True' if x > 0 else 'False')
# Combine related features to reduce the number of features the model considers and to remove correlation between predictors. 



df_full['YrBlt_Remod']=df_full['YearBuilt']+df_full['YearRemodAdd']

df_full['Total_area']=df_full['TotalBsmtSF'] + df_full['1stFlrSF'] + df_full['2ndFlrSF']



df_full['Total_finished'] = (df_full['BsmtFinSF1'] + df_full['BsmtFinSF2'] +

                                 df_full['1stFlrSF'] + df_full['2ndFlrSF'])



df_full['Total_Bathrooms'] = (df_full['FullBath'] + (0.5 * df_full['HalfBath']) + # *0.5 for half bathrooms

                               df_full['BsmtFullBath'] + (0.5 * df_full['BsmtHalfBath']))



df_full['Total_porch'] = (df_full['OpenPorchSF'] + df_full['3SsnPorch'] +

                              df_full['EnclosedPorch'] + df_full['ScreenPorch'] +

                              df_full['WoodDeckSF'])



# Remove the features that have been combined in the new features



df_full = df_full.drop(['YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF', '2ndFlrSF','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'], axis=1)
# Create dummy variables from categoricals. 

for feature in df_full.select_dtypes(include=['object']).columns: # do it for categorical columns only

    replace=df_full.loc[:,feature].value_counts().idxmax()

    df_full[feature]=df_full.loc[:,feature].fillna(replace)

df_full = pd.get_dummies(df_full).reset_index(drop=True)

print(df_full.shape)

print(df_full.info())
# Visualise the original distribution of the target variable using a histogram with a trend line.

sns.distplot(target_var, bins = 30, )

plt.title('Sale Price before normalisation')



# Print the measure of Skewness and Kurtosis for the target variable before transformation. 

print("Skewness: %f" % target_var.skew())

print("Kurtosis: %f" % target_var.kurt())
# log transform target variable.

target_normal = np.log1p(target_var)



# Visualise normality of the normalised target. 

sns.distplot(target_normal, bins = 30, )

plt.title('Normalised Sale Price')

# Print the measure of Skewness and Kurtosis for the target variable post transformation.

print("Skewness: %f" % target_normal.skew())

print("Kurtosis: %f" % target_normal.kurt())
# Measures of skewness and kurtosis for numeric features in the full dataset prior to transformation. 

for numeric_feature in df_full.select_dtypes(exclude=['object']):

    print(numeric_feature)

    print("Skewness: %f" % df_full[numeric_feature].skew())

    print("Kurtosis: %f" % df_full[numeric_feature].kurt())

    print(" ")
# Then we create a bar plot of the measures of skewness for each numeric feature within the training set before transformation.

plt.figure(figsize=(15,5))



numerical_features = df_train.select_dtypes(exclude=['object'])

# Once again we compute skewness for all possible predictors. 

numerical_skewness = numerical_features.apply(lambda x: skew(x)) 

np.abs(numerical_skewness).plot.bar()

plt.title("Skewness calculated on original training set's numeric features")

plt.ylabel("Skewness")



#Can observe skewness at two different thresholds. 



#numerical_skewness = numerical_skewness[numerical_skewness > 0.75]

numerical_skewness = numerical_skewness[numerical_skewness > 1]



# Then we create a bar plot of the measures of skewness for each numeric feature within the training set post transformation.

plt.figure(figsize=(15,5))





numerical_features = df_train.select_dtypes(exclude=['object'])

# Once again we compute skewness for all possible predictors once they have been transformed. 

numerical_skewness = numerical_features.apply(lambda x: skew(np.log1p(x))) 

np.abs(numerical_skewness).plot.bar()

plt.title("Skewness calculated on original training set's numeric features once log transformed")

plt.ylabel("Skewness")



#Can observe skewness at two different thresholds. 



#numerical_skewness = numerical_skewness[numerical_skewness > 0.75]

numerical_skewness = numerical_skewness[numerical_skewness > 1]
# log transform numeric predictors within full dataset



numerical_features = df_full.select_dtypes(exclude=['object'])

numerical_skewness = numerical_features.apply(lambda x: skew(x))

numerical_skewness  = numerical_skewness [numerical_skewness  > 0.75]

#Can observe skewness at two different thresholds. 

#numerical_skewness = numerical_skewness[numerical_skewness > 1]

numerical_skewness = numerical_skewness.index

df_full_norm = df_full.copy()

df_full_norm[numerical_skewness] = np.log1p(df_full_norm[numerical_skewness])
# Measures of skewness and kurtosis for numeric features in the full dataset post log transformation. 

for numeric_feature in df_full_norm.select_dtypes(exclude=['object']):

    print(numeric_feature)

    print("Skewness: %f" % df_full_norm[numeric_feature].skew())

    print("Kurtosis: %f" % df_full_norm[numeric_feature].kurt())

    print(" ")
# Separation of dataset back to original groups based on boolean column we created before concatenating the datasets. 



df_train_norm = df_full_norm[df_full_norm['training_set']==True]

df_test_norm = df_full_norm[df_full_norm['training_set']==False]



# We then drop the columns that we originally created to distinguish them. 



df_train_norm = df_train_norm.drop('training_set', axis=1)

df_test_norm = df_test_norm.drop('training_set', axis=1)
# Train and test subsets from original training set. 

X_train, X_test, y_train, y_test = train_test_split(df_train_norm, target_normal, test_size=0.2, random_state=42)
# We create an object from the random forest regressor class and fit it to our transformed dataset. 

rf_norm = RandomForestRegressor(n_estimators=100, n_jobs=-1)

rf_norm.fit(df_train_norm, target_normal)



# We use cross validation to optimise the scores.  

cv_num=4

scores = cross_val_score(rf_norm, df_train_norm, target_normal, cv=cv_num, scoring='neg_mean_squared_error')

print('Average root mean squared log error (Transformed dataset) =', np.mean(np.sqrt(-scores)))



# Now we can make predictions on the normalised dataset,

#as well as reverting the data to its original state prior to transformation by using the exponential function.

preds_norm = np.expm1(rf_norm.predict(df_test_norm))
# Create plot to compare the distributions for the actual y_test target values, to those predicted by the random forest regressor. 



plt.hist(preds_norm, label = "Predictions")

plt.hist(target_var, alpha = 0.5, label = 'Target variable')

plt.title("Predicted values using the random forest regressor versus target variable")

plt.legend()

plt.show()

# First we create an object from the LassoCV class. 

lasso = LassoCV(alphas = np.logspace(-4, 10, 2), cv = 4, n_jobs=4, max_iter=1000000000) # Very high number of iteration. 

# Fit the regressor object to the transformed dataset and target variable.

lasso.fit(df_train_norm, target_normal)



print('average root mean squared log error (lasso)=', np.mean(np.sqrt(-scores)))

# Transorm the predictions made back to the original state before the log transformation. 

preds_lasso = np.expm1(lasso.predict(df_test_norm))

# Create plot to compare the distributions for the actual y_test target values, to those predicted by the Lasso regressor. 



plt.hist(preds_lasso, label = "Predictions")

plt.hist(target_var, alpha = 0.5, label = 'Target variable')

plt.title("Predicted values using LassoCV versus target variable")

plt.legend()

plt.show()



# Create XGBRegressor object

xgb = XGBRegressor(colsample_bytree=0.02, learning_rate=0.05, max_depth=3, n_estimators=1920)

# Fit too transformed dataset          

xgb.fit(df_train_norm, target_normal)



scores = cross_val_score(rf_norm, df_train_norm, target_normal, cv=cv_num, scoring='neg_mean_squared_error')

print('average root mean squared log error (xgboost)=', np.mean(np.sqrt(-scores)))

preds_xgb = np.expm1(xgb.predict(df_test_norm))

# Create plot to compare the distributions for the actual y_test target values, to those predicted by the XGBoost regressor. 



plt.hist(preds_xgb, label = "Predictions")

plt.hist(target_var, alpha = 0.5, label = 'Target variable')

plt.title("Predicted values using XGBoost versus target variable")

plt.legend()

plt.show()
preds_avg=(preds_lasso+preds_xgb)/2

Group_5 = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds_avg})

Group_5.to_csv('submission_FINAL.csv', index=False)