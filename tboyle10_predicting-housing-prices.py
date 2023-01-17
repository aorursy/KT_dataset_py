import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


# setting styles for plotting
plt.rcParams['figure.figsize'] = [10, 5]
sns.set_palette('husl')
sns.set_style('whitegrid')
sns.set_context('talk')
df = pd.read_csv('../input/Melbourne_housing_FULL.csv')

print(df.shape)
df.head()
df.columns
# 1st things 1st - we're trying to predict value - can't do that with nan values for price
#dropping rows where price = nan
df = df[np.isfinite(df.Price)]
df.head()
df.describe()
# checking for outliers - 75% of homes have 4 rooms and the max is 16
df.loc[df['Rooms'] > 10]
# car seems like it has some outliers ... 
df.loc[df['Car'] > 10]
df.info()
#checking for missing values

# from https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb# from  
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values_table(df).head(10)
def drop_missing_values(df, percent_drop):
    """
    Drop columns with missing values.
    
    Args:
        df = dataframe
        percent_drop = percentage of null values above which the column will be dropped
            as decimal between 0 and 1
    Returns:
        df = df where columns above percent_drop are dropped.
    
    """
    to_drop = [column for column in df if (df[column].isnull().sum()/len(df) >= percent_drop)]

    print('Columns to drop: ' , (len(to_drop)))
    # Drop features 
    df = df.drop(columns=to_drop)
    print('Shape: ', df.shape)
    return df
#dropping columns where >60% of values missing
df = drop_missing_values(df, .6)
# find correlations to target = price
corr_matrix = df.corr().abs()

print(corr_matrix['Price'].sort_values(ascending=False).head(20))
# Visualizing the correlation matrix
# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
sns.heatmap(upper)
plt.show();
#dropping highly correlated features
#code from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

# Find index of feature columns with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
print('Columns to drop: ' , (len(to_drop)))

# Drop features 
df = df.drop(columns=to_drop)
print('train_features_df shape: ', df.shape)
# plotting distribution of price ... 
# The long right tail indicates there are probably highly priced outliers
sns.distplot(df.Price);
#checking data types of columns
df.dtypes
# checking for categorical variables
df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
df.shape
# Address has a very high percentage of unique values
# Theoretically each home has a different address
# Duplicated are being resold possibly
# But bottom line I'm assuming address won't be valuable in the model so dropping the column

df = df.drop(['Address'], 1)
# visualizing some categorical variables relationship to price

plt_data = df[['Price', 'Suburb', 'Type', 'Method', 'Date', 'CouncilArea', 'Regionname']].copy()
plt_data = plt_data.dropna()

fig, ax = plt.subplots(figsize=(20,5))

fig.add_subplot(121)
sns.boxplot(x=plt_data.Type, y=plt_data.Price)

fig.add_subplot(122)
sns.stripplot(x=plt_data.Method, y=plt_data.Price)

plt.show;
sns.boxplot(x=plt_data.Regionname, y=plt_data.Price)
plt.xticks(rotation=90);
# I don't think the seller should have anything to do with the home price,,,
# but plotting just to be sure
fig, ax = plt.subplots()

plt.scatter(df.SellerG, df.Price)
ax.set_xticks([])
plt.show();
# dropping column SellerG
df = df.drop('SellerG', axis=1)
#Checking the Date column
fig, ax = plt.subplots()

plt.scatter(df.Date, df.Price)
ax.set_xticks([])
plt.show();
# Date also doesn't seem like it will be useful so dropping
df = df.drop('Date', axis=1)
# checking categorical variables again
df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
#plotting postcode vs price
#it definitely looks like certain postcodes are more popular and expensive
plt.scatter(df.Postcode, df.Price);
# converting postcode to categorical
# set up bins
# just setting up random bins by looking at the plot above
bins = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,]
category = pd.cut(df.Postcode, bins)
category = category.to_frame()
df['Postcode'] = category
# and checking to make sure it worked

sns.barplot(x=df.Postcode, y=df.Price)
plt.xticks(rotation=90);
# one hot encoding categorical variables
df = pd.get_dummies(df)
# and checking to make sure it worked
df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
# Now finding the outliers - starting with the really expensive homes
df.sort_values('Price', ascending=False).head()
# dropping the home with the 11 million dollar price
df = df.drop(25635)
# checking for really inexpensive homes
df.sort_values('Price', ascending=True).head()
# dropping index 4378
#df = df.drop(4378)
# we need to deal with missing values before modeling
missing_values_table(df)
# dropping YearBuilt since over 50% of values missing
df = df.drop(['YearBuilt'], 1)
# seperating our target out from the dataframe
price = df.Price
df = df.drop(['Price'], 1)
# imputing values for remaining null values
imputer = Imputer(strategy='mean')

imputed = imputer.fit_transform(df)
# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(imputed, price, test_size=0.25, random_state=27)
# normalizing the data
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_train = scaler.fit_transform(X_train)
from sklearn import ensemble

GBR = ensemble.GradientBoostingRegressor().fit(X_train, y_train)
y_ = GBR.predict(X_test)
y_
GBR.score(X_test, y_test)
# checking which are the most important features
feature_importance = GBR.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[-20:-1:1]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
GBR2 = ensemble.GradientBoostingRegressor(max_depth=5, n_estimators=500, min_samples_split=5).fit(X_train, y_train)
GBR2.score(X_test, y_test)
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_))
print('MSE:',metrics.mean_squared_error(y_test,y_))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_)))
# checking which are the most important features
feature_importance = GBR2.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[-20:-1:1]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
