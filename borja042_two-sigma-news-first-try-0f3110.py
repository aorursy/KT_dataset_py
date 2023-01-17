# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime 

import sklearn # ML

#from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from kaggle.competitions import twosigmanews

# Any results you write to the current directory are saved as output.
# Retreive the environment of the competition
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Data loaded!')
# Retrieve all training data
(market_train_df, news_train_df) = env.get_training_data()
print("Fetching training data finished... ")
print('Data obtained!')
# Preprocessing
# Reorder universe so its right after assetName
# cols=market_train_df.columns.tolist()
# cols=cols[:3]+[cols[-1]]+cols[3:-1]
# market_train_df = market_train_df[cols]
# Remove universe, as it doesnt exist on the pred data
market_train_df.drop(['universe'], axis=1, inplace=True)
# Adding daily difference
new_col = market_train_df["close"] - market_train_df["open"]
market_train_df.insert(loc=6, column="daily_diff", value=new_col)
# Market data analysis
# Types of the columns
print(market_train_df.dtypes)
market_train_df.head()

print(news_train_df.dtypes)
print(news_train_df.head())
print(news_train_df.loc[2,'subjects'], type(news_train_df.loc[2,'subjects']))
news_train_df.head()
news_train_df['subjects'].astype(list)
# Example of the columns
market_train_df.head()
# Variables description
market_train_df.describe()
market_train_df.isnull().sum()
# Correlation between the numericals (except universe)
# Note that this removes the null values from the computation
market_train_df.iloc[:, 3:].corr(method='pearson')
# Lets analyze the assets
assets=market_train_df["assetCode"].unique()
print(len(assets))
print(assets)
# Lets analyze further the target variable
# Very big outliers, lets see their number and distribution
fig, axes = plt.subplots(3,2, figsize=(20, 12)) # create figure and axes
print("# Rows with |value| > 1 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>1].shape[0])
print("# Rows with |value| > 0.5 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>0.5].shape[0])
print("# Rows with |value| > 0.25 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>0.25].shape[0])
print("# Rows with |value| > 0.1 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>0.1].shape[0])

# Boxplot with all values
market_train_df.boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[0])
axes.flatten()[0].set_xlabel('Boxplot with all values', fontsize=18)
# Removing rows with outliers (bigger or smaller than 1)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<1].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[1])
axes.flatten()[1].set_xlabel('Boxplot with values such that |val| < 1', fontsize=18)
# Removing rows with outliers (bigger or smaller than 0.5)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.5].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[2])
axes.flatten()[2].set_xlabel('Boxplot with values such that |val| < 0.5', fontsize=18)
# Removing rows with outliers (bigger or smaller than 0.25)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.25].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[3])
axes.flatten()[3].set_xlabel('Boxplot with values such that |val| < 0.25', fontsize=18)
# Removing rows with outliers (bigger or smaller than 0.1)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.1].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[4])
axes.flatten()[4].set_xlabel('Boxplot with values such that |val| < 0.1', fontsize=18)
# Distribution of the target value (not including values bigger or smaller than 1)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.25].hist(column="returnsOpenNextMktres10", bins=100, ax=axes.flatten()[5])
axes.flatten()[5].set_xlabel('Histogram for values such that |val| < 0.25', fontsize=18)
print("The variable is actually centered in 0 and only a few outliers higher than 0.1. This makes sense considering that the returns of the \
market for 10 days are really small. Our goal then should be to detect those times in which the wins or loses are really high by making \
use of the news. A good approach for this could be an algorithm to control the small temporal oscilation of the market and then use the news \
to detect those imprevisible changes.")

# Number of null values
market_train_df.isna().sum()
# Where are those null values?
rows_with_null=market_train_df[pd.isnull(market_train_df).any(axis=1)]
dates_with_null=rows_with_null["time"].unique()
nulls_per_date=[rows_with_null[rows_with_null["time"]==d].shape[0] for d in dates_with_null]
caca=pd.DataFrame({'date': dates_with_null, 'nulls': nulls_per_date })
caca
# Where are those null values?
rows_with_null=market_train_df[pd.isnull(market_train_df).any(axis=1)]
assets_with_null=rows_with_null["assetCode"].unique()
nulls_per_asset=[rows_with_null[rows_with_null["assetCode"]==a].shape[0] for a in assets_with_null]
caca=pd.DataFrame({'asset': assets_with_null, 'nulls': nulls_per_asset})
caca.sort_values(by=['nulls'], ascending=False, inplace=True)
caca
# Possibilities to deal with missing values:
# Remove the rows with missing values
# We cannot follow this approach to predict, just to be sure while training
#market_train_df.dropna(inplace=True)
# Remove the cols with missing values
# We cannot follow this approach to predict
# What if the cols with missing vals are different? Or if those are relevant?
# market_train_df.dropna(inplace=True, axis=1)
# Fill in with shitty values such as -99999
# Tricky, but model can learn it, overall decission trees
# ... we will do this with sklearn
# market_train_df.isna().sum()
market_train_df.isnull().sum()
nuls = market_train_df.loc[:,['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']]
#print (nuls)
tol = 0.10
nuls_r1 = market_train_df.loc[market_train_df['returnsClosePrevMktres1'] <= tol,'returnsClosePrevMktres1']
print(nuls_r1, len(nuls_r1), 'Media--> ', nuls_r1.mean())
# Vamos a cambiar los nuls por las medias quitando outliers para esa media.
tol = 0.15
lista_valores_rellenar = list()
for col in nuls:
    print (nuls[col], "Media--> ", nuls[col].loc[nuls[col]<= tol].mean())
    lista_valores_rellenar += [nuls[col].loc[nuls[col]<= tol].mean()]
lista_valores_rellenar
# Rellenamos 
#market_train_df.loc[[:,['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']]
market_train_df.loc[pd.Series(market_train_df['returnsClosePrevMktres1'].isnull()), 'returnsClosePrevMktres1'] = lista_valores_rellenar[0]
market_train_df.loc[pd.Series(market_train_df['returnsOpenPrevMktres1'].isnull()), 'returnsOpenPrevMktres1'] = lista_valores_rellenar[1]
market_train_df.loc[pd.Series(market_train_df['returnsClosePrevMktres10'].isnull()), 'returnsClosePrevMktres10'] = lista_valores_rellenar[2]
market_train_df.loc[pd.Series(market_train_df['returnsOpenPrevMktres10'].isnull()), 'returnsOpenPrevMktres10'] = lista_valores_rellenar[3]
print(lista_valores_rellenar)


print(market_train_df.isnull().sum())

# News data analysis. Habría que juntar por fecha(hasta día y por assetName)
print(len(market_train_df.loc[pd.Series(market_train_df['assetName'] == 'Agilent Technologies Inc'),:]))
len(market_train_df.loc[pd.Series(market_train_df['assetCode'] == 'A.N'),:])

print(len(news_train_df.loc[pd.Series(news_train_df['assetCodes'] == 'A.N'),:]))
print(len(news_train_df.loc[pd.Series(news_train_df['assetName'] == 'Agilent Technologies Inc'),:]))


# Las columnas dailidiff y assetcode no aportan mucho
#market_train_df = market_train_df.drop(['daily_diff', 'assetCode'], axis = 1)
market_train_df.loc[pd.Series(market_train_df['assetName'] == 'Agilent Technologies Inc'),:].head()
# Para el market train los datos no se repiten. Habría que ver si hay varias noticias para el mismo asset en un día.
# sourceTimestamp y firstCreated no aporta na
news_train_df = news_train_df.drop(['sourceTimestamp','firstCreated'], axis = 1)
news_train_df.loc[pd.Series(news_train_df['assetName'] == 'Agilent Technologies Inc'),:].head()



# La unión de los datasets. Habrá que juntar por día y asset code. Para ello recortamos las fechas. Si hay varias noticias en un día
#se juntarán tmbién
news_train_df['time2'] = news_train_df['time'].astype(str).str.slice(0,10)
market_train_df['time2'] = market_train_df['time'].astype(str).str.slice(0,10)


news_train_df.loc[pd.Series(news_train_df['assetName'] == 'Agilent Technologies Inc'),:].head()
market_train_df.loc[pd.Series(market_train_df['assetName'] == 'Agilent Technologies Inc'),:].head()
df_final = market_train_df.merge(news_train_df, how = 'left', on = ['assetName', 'time2'])

df_final.head()
df_final.loc[abs(df_final['returnsOpenNextMktres10']) <0.25, 'returnsOpenNextMktres10'].hist(bins = 10000)
# Toy prediction example
# Remove outliers to make some tests
F = market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.25]
# Imputer to remove nans
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-9999.99)
T = pd.DataFrame(imp.fit_transform(F), columns=F.columns)
T
# Define data to use for X and y
n = 1000000
X = T.iloc[:n, 3:-1]
y = T[T.columns[-1]][:n]
print(X.shape, y.shape)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# Save cols order for the prediction data
cols_order=X_train.columns

# Predict
regr = RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=0, n_estimators=20)
regr.fit(X_train, y_train)
y_predicted=regr.predict(X_test)
print("Acabó")
mean_absolute_error(y_test, y_predicted)
for i in range(X.shape[1]):
    print("%s (%f)" % (X.columns[i], regr.feature_importances_[i]))
df_results = X_test
df_results.insert(loc=df_results.shape[1], column="y_real", value=y_test)
df_results.insert(loc=df_results.shape[1], column="y_pred", value=y_predicted)
df_results.head()


# Retrieve all days to iterate through
# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
def rfr_predictions(market, news, predictions_template_df):
    print(market["time"][0])
    copy=market.copy()
    # Adding daily difference
    new_col = copy["close"] - copy["open"]
    copy.insert(loc=6, column="daily_diff", value=new_col)
    # Getting columns used on the training only and reorder
    copy=copy[cols_order]
    # Replacing missing values
    copy = pd.DataFrame(imp.fit_transform(copy), columns=copy.columns)
    # Predicting
    y_predicted=regr.predict(copy)
    mn=min(y_predicted)
    mx=max(y_predicted)
    # Converting into the confidence value, from -1 to 1
    predictions_template_df.confidenceValue = [((y-(-0.25))/(0.25-(-0.25))*2-1) for y in y_predicted]
# Generate the predictions
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    rfr_predictions(market_obs_df, news_obs_df, predictions_template_df)
    env.predict(predictions_template_df)
print('Prediction finished!')
#env.predict(predictions_template_df)

# Write submission file
# Note that for submitting the results we have to commit and then upload the resulting csv file
env.write_submission_file()
print([filename for filename in os.listdir('.') if '.csv' in filename])

