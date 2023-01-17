import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Import our libraries we are going to use for our data analysis.
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Plotly visualizations
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# For oversampling Library (Dealing with Imbalanced Datasets)
from imblearn.over_sampling import SMOTE
from collections import Counter
from IPython.display import HTML
import warnings; warnings.simplefilter('ignore')

data_file = "..//input/train.csv"
train = pd.read_csv(data_file,low_memory=False, index_col=0)
train.dtypes
data_file = "..//input/test.csv"
test = pd.read_csv(data_file,low_memory=False, index_col=0)
test.shape,train.shape
# multiple aggregation functions can be applied simultaneously
stat1=train.groupby('BldgType').SalePrice.agg(['count', 'mean', 'min', 'max'])
df1 = pd.DataFrame(stat1)
df1
a= df1.plot(kind='bar',title='Type of dwelling VS Sales/Price')
# multiple aggregation functions can be applied simultaneously
stat2=train.groupby('YearBuilt').SalePrice.agg(['count', 'mean', 'min', 'max'])
df2 = pd.DataFrame(stat2)
df2
a= df2.plot(title='Year Build with Sales/Price')
# multiple aggregation functions can be applied simultaneously
stat2=train.groupby('OverallQual').SalePrice.agg(['count', 'mean', 'min', 'max'])
df2 = pd.DataFrame(stat2)
df2
a= df2.plot(title='Overall Quality VS Sales/Price')
# multiple aggregation functions can be applied simultaneously
stat2=train.groupby('OverallCond').SalePrice.agg(['count', 'mean', 'min', 'max'])
df2 = pd.DataFrame(stat2)
df2
a= df2.plot(title='Overall Condition VS Sales/Price')
# multiple aggregation functions can be applied simultaneously
stat2=train.groupby('HouseStyle').SalePrice.agg(['count', 'mean', 'min', 'max'])
df2 = pd.DataFrame(stat2)
df2
a= df2.plot(kind='bar', title='HouseStyle VS Sales/Price')
# multiple aggregation functions can be applied simultaneously
stat2=train.groupby('LotArea').SalePrice.agg(['count', 'mean', 'min', 'max'])
df2 = pd.DataFrame(stat2)
df2
# Property by size and Sale Price Score grade

sns.set_style('whitegrid')

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
cmap = plt.cm.inferno

by_year_lot = train.groupby(['YearBuilt', 'LotArea']).SalePrice.mean()
by_year_lot.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, legend=False, ax=ax1, figsize=(16,12))
ax1.set_title('Average Sale Price by Year Built and Lot Area', fontsize=14)

by_HouseStyle = train.groupby(['ExterQual','HouseStyle']).SalePrice.mean()
by_HouseStyle.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, legend=False, ax=ax2, figsize=(16,12))
ax2.set_title('Average Sale Price by House Style and External Quality', fontsize=14)

#OverallCond

by_OverallCond = train.groupby(['LotArea', 'OverallCond']).SalePrice.mean()
by_OverallCond.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, legend=False, ax=ax3, figsize=(16,12))
ax3.set_title('Average Sale Price by Overall Condition and Lot Area', fontsize=14)

by_Neighborhood = train.groupby(['BldgType','HouseStyle','Neighborhood']).SalePrice.mean()
by_Neighborhood.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, ax=ax4, figsize=(16,12))
ax4.set_title('Average Sale Price by House Style,BldgType,Neighborhood ', fontsize=14)
ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)
fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(20,12))


sns.violinplot(x="HouseStyle", y="Fireplaces", data=train, palette="Set2", ax=ax1)
sns.violinplot(x="HouseStyle", y="BedroomAbvGr", data=train, palette="Set2", ax=ax2)
sns.boxplot(x="HouseStyle", y="GarageArea", data=train, palette="Set2", ax=ax3)
sns.boxplot(x="HouseStyle", y="KitchenAbvGr", data=train, palette="Set2", ax=ax4)
fig, ax = plt.subplots(1, 3, figsize=(16,5))




SalePrice = train["SalePrice"].values
YrSold = train["YrSold"].values
MoSold = train["MoSold"].values


sns.distplot(SalePrice, ax=ax[0], color="#F7522F")
ax[0].set_title("Property Sold by Price", fontsize=14)
sns.distplot(YrSold, ax=ax[1], color="#2F8FF7")
ax[1].set_title("Property Sold by Year", fontsize=14)
sns.distplot(MoSold, ax=ax[2], color="#2EAD46")
ax[2].set_title("Property Sold by month", fontsize=14)
plt.figure(figsize=(20,8))
sns.barplot('HouseStyle', 'SalePrice', data=train, palette='tab10')
plt.title('SalePrice', fontsize=16)
plt.xlabel('House Style', fontsize=14)
plt.ylabel('Average Sale Price', fontsize=14)
# Determining the property that are bad from verallCond column

bad_property = [1,2,3,4,5]


train['property_condition'] = np.nan

def property_condition(OverallCond):
    if OverallCond in bad_property:
        return '1'
    else:
        return '0'        
    
    
train['property_condition'] = train['OverallCond'].apply(property_condition)
f, ax = plt.subplots(1,2, figsize=(20,10))

colors = ["#3791D7", "#D72626"]
labels ="Good condition", "Bad condition"

plt.suptitle('Information on Dwelling Conditions', fontsize=20)

train["property_condition"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)

#ax[0].set_ylabel('% of Condition of Dwelling', fontsize=14)

palette = ["#3791D7", "#D72626"]

sns.barplot(x="YrSold", y="SalePrice", hue="property_condition", data=train, palette=palette, estimator=lambda x: len(x) / len(train) * 100)
ax[1].set(ylabel="(%)")
plt.style.use('dark_background')
cmap = plt.cm.Set3

by_Neighborhood = train.groupby(['YrSold', 'MSZoning']).SalePrice.sum()
by_Neighborhood.unstack().plot(stacked=False, colormap=cmap, grid=False, legend=True, figsize=(15,6))

plt.title('Property sold by Type of Property', fontsize=16)
#### Most demanded 
plt.style.use('dark_background')
cmap = plt.cm.Set3

by_Neighborhood = train.groupby(['YrSold', 'Neighborhood']).SalePrice.sum()
by_Neighborhood.unstack().plot(stacked=False, colormap=cmap, grid=False, legend=True, figsize=(16,7))

plt.title('Property sold Neighborhood', fontsize=20)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x="YrSold", y="SalePrice", data=train)
sns.lmplot(x="YrSold", y="SalePrice", hue="property_condition", data=train);
sns.lmplot(x="YrSold", y="SalePrice", col="Neighborhood", data=train, col_wrap=3)
sns.lmplot(x="YrSold", y="SalePrice", col="MSZoning", data=train,
           aspect=.5);
sns.lmplot(x="YrSold", y="SalePrice", col="HouseStyle", data=train, col_wrap=3)
sns.jointplot(x="YrSold", y="SalePrice", data=train, kind="reg");
train.iloc[0:10, 1:15]
data_file = "..//input/train.csv"
train = pd.read_csv(data_file,low_memory=False, index_col=0)
train.shape
train.isnull().sum()
train.isnull().sum()
# fill in missing values with a specified value
train['BsmtQual'].fillna(value='TA', inplace=True)
train.BsmtQual.isnull().sum()
train.shape
train.dtypes
train.isnull().sum()
# fill in missing values with a specified value
train['GarageQual'].fillna(value='TA', inplace=True)
# fill in missing values with a specified value
train['GarageCond'].fillna(value='TA', inplace=True)
# fill in missing values with a specified value
train['BsmtCond'].fillna(value='TA', inplace=True)
train.GarageCond.mode()
train.isnull().sum()
train["MSZoning_cat"] = train["MSZoning"].astype('category')
train["MSZoning_num"] = train["MSZoning_cat"].cat.codes
train.drop('MSZoning_cat', axis=1, inplace=True)
train["MSZoning_num"].unique()
train["Street_cat"] = train["Street"].astype('category')
train["Street_num"] = train["Street_cat"].cat.codes
train.drop('Street_cat', axis=1, inplace=True)
train["Street_num"].unique()
train.dtypes
train["SaleCondition_cat"] = train["SaleCondition"].astype('category')
train["SaleCondition_num"] = train["SaleCondition_cat"].cat.codes
train.drop('SaleCondition_cat', axis=1, inplace=True)
train["SaleType_cat"] = train["SaleType"].astype('category')
train["SaleType_num"] = train["SaleType_cat"].cat.codes
train.drop('SaleType_cat', axis=1, inplace=True)
train["GarageCond_cat"] = train["GarageCond"].astype('category')
train["GarageCond_num"] = train["GarageCond_cat"].cat.codes
train.drop('GarageCond_cat', axis=1, inplace=True)
train["GarageQual_cat"] = train["GarageQual"].astype('category')
train["GarageQual_num"] = train["GarageQual_cat"].cat.codes
train.drop('GarageQual_cat', axis=1, inplace=True)
train["Functional_cat"] = train["Functional"].astype('category')
train["Functional_num"] = train["Functional_cat"].cat.codes
train.drop('Functional_cat', axis=1, inplace=True)
train["BsmtCond_cat"] = train["BsmtCond"].astype('category')
train["BsmtCond_num"] = train["BsmtCond_cat"].cat.codes
train.drop('BsmtCond_cat', axis=1, inplace=True)
train["BsmtQual_cat"] = train["BsmtQual"].astype('category')
train["BsmtQual_num"] = train["BsmtQual_cat"].cat.codes
train.drop('BsmtQual_cat', axis=1, inplace=True)
train["Foundation_cat"] = train["Foundation"].astype('category')
train["Foundation_num"] = train["Foundation_cat"].cat.codes
train.drop('Foundation_cat', axis=1, inplace=True)
train["ExterCond_cat"] = train["ExterCond"].astype('category')
train["ExterCond_num"] = train["ExterCond_cat"].cat.codes
train.drop('ExterCond_cat', axis=1, inplace=True)
train["RoofStyle_cat"] = train["RoofStyle"].astype('category')
train["RoofStyle_num"] = train["RoofStyle_cat"].cat.codes
train.drop('RoofStyle_cat', axis=1, inplace=True)
train["HouseStyle_cat"] = train["HouseStyle"].astype('category')
train["HouseStyle_num"] = train["HouseStyle_cat"].cat.codes
train.drop('HouseStyle_cat', axis=1, inplace=True)
train["Utilities_cat"] = train["Utilities"].astype('category')
train["Utilities_num"] = train["Utilities_cat"].cat.codes
train.drop('Utilities_cat', axis=1, inplace=True)
train["Neighborhood_cat"] = train["Neighborhood"].astype('category')
train["Neighborhood_num"] = train["Neighborhood_cat"].cat.codes
train.drop('Neighborhood_cat', axis=1, inplace=True)
train.dtypes
# create a list of features
feature_cols = ['MSZoning_num',          
'Street_num',             
'Utilities_num',          
'Neighborhood_num',       
'HouseStyle_num',         
'RoofStyle_num',          
'ExterCond_num',          
'Foundation_num',         
'BsmtQual_num',           
'BsmtCond_num',           
'Functional_num',         
'GarageQual_num',         
'GarageCond_num',         
'SaleType_num',           
'SaleCondition_num',      
'PoolArea',              
'MiscVal',               
'MoSold',                
'YrSold',                
'Fireplaces',            
'GarageArea',            
'FullBath',              
'HalfBath',              
'BedroomAbvGr',          
'TotRmsAbvGrd',          
'YearRemodAdd',          
'LotArea']
X = train[feature_cols]
y = train.SalePrice
X = train[feature_cols]
y = train.SalePrice
# import class, instantiate estimator, fit with all data
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
rfclf.fit(train[feature_cols], train.SalePrice)
# compute the feature importances
a = pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})
model = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
model.fit(X, y)

feature_importance = model.feature_importances_
feature_importance = rfclf.feature_importances_
features = feature_cols
plt.figure(figsize=(16, 6))
plt.yscale('log', nonposy='clip')

plt.bar(range(len(feature_importance)), feature_importance, align='center')
plt.xticks(range(len(feature_importance)), features, rotation='vertical')
plt.title('Feature importance')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()

data_file = "..//input/sample_submission.csv"
left = pd.read_csv(data_file,low_memory=False, index_col=0)
data_file = "..//input/test.csv"
right = pd.read_csv(data_file,low_memory=False, index_col=0)
right.dtypes
right.dtypes
df = pd.merge(left, right, how='inner', on=['Id'])
df.isnull().sum()
# fill in missing values with a specified value
df['BsmtQual'].fillna(value='TA', inplace=True)
# fill in missing values with a specified value
df['GarageQual'].fillna(value='TA', inplace=True)
# fill in missing values with a specified value
df['GarageCond'].fillna(value='TA', inplace=True)
# fill in missing values with a specified value
df['BsmtCond'].fillna(value='TA', inplace=True)
df.MSZoning.mode()
df['MSZoning'].fillna(value='RL', inplace=True)
df.Utilities.mode()
df['Utilities'].fillna(value='AllPub', inplace=True)
df.GarageArea.mean()
df['GarageArea'].fillna(value='472.76', inplace=True)
df.SaleType.mode()
df['SaleType'].fillna(value='WD', inplace=True)
df.isnull().sum()
df["MSZoning_cat"] = df["MSZoning"].astype('category')
df["MSZoning_num"] = df["MSZoning_cat"].cat.codes
df.drop('MSZoning_cat', axis=1, inplace=True)
df["Street_cat"] = df["Street"].astype('category')
df["Street_num"] = df["Street_cat"].cat.codes
df.drop('Street_cat', axis=1, inplace=True)
df["SaleCondition_cat"] = df["SaleCondition"].astype('category')
df["SaleCondition_num"] = df["SaleCondition_cat"].cat.codes
df.drop('SaleCondition_cat', axis=1, inplace=True)
df["SaleType_cat"] = df["SaleType"].astype('category')
df["SaleType_num"] = df["SaleType_cat"].cat.codes
df.drop('SaleType_cat', axis=1, inplace=True)
df["GarageCond_cat"] = df["GarageCond"].astype('category')
df["GarageCond_num"] = df["GarageCond_cat"].cat.codes
df.drop('GarageCond_cat', axis=1, inplace=True)
df["GarageQual_cat"] = df["GarageQual"].astype('category')
df["GarageQual_num"] = df["GarageQual_cat"].cat.codes
df.drop('GarageQual_cat', axis=1, inplace=True)
#df["Functional_cat"] = df["Functional"].astype('category')
#df["Functional_num"] = df["Functional_cat"].cat.codes
#df.drop('Functional_cat', axis=1, inplace=True)
df["BsmtCond_cat"] = df["BsmtCond"].astype('category')
df["BsmtCond_num"] = df["BsmtCond_cat"].cat.codes
df.drop('BsmtCond_cat', axis=1, inplace=True)
df["BsmtQual_cat"] = df["BsmtQual"].astype('category')
df["BsmtQual_num"] = df["BsmtQual_cat"].cat.codes
df.drop('BsmtQual_cat', axis=1, inplace=True)
df["Foundation_cat"] = df["Foundation"].astype('category')
df["Foundation_num"] = df["Foundation_cat"].cat.codes
df.drop('Foundation_cat', axis=1, inplace=True)
df["ExterCond_cat"] = df["ExterCond"].astype('category')
df["ExterCond_num"] = df["ExterCond_cat"].cat.codes
df.drop('ExterCond_cat', axis=1, inplace=True)
df["RoofStyle_cat"] = df["RoofStyle"].astype('category')
df["RoofStyle_num"] = df["RoofStyle_cat"].cat.codes
df.drop('RoofStyle_cat', axis=1, inplace=True)
df["HouseStyle_cat"] = df["HouseStyle"].astype('category')
df["HouseStyle_num"] = df["HouseStyle_cat"].cat.codes
df.drop('HouseStyle_cat', axis=1, inplace=True)
df["Utilities_cat"] = df["Utilities"].astype('category')
df["Utilities_num"] = df["Utilities_cat"].cat.codes
df.drop('Utilities_cat', axis=1, inplace=True)
df["Neighborhood_cat"] = df["Neighborhood"].astype('category')
df["Neighborhood_num"] = df["Neighborhood_cat"].cat.codes
df.drop('Neighborhood_cat', axis=1, inplace=True)
df.dtypes
df['SalePrice'] = df['SalePrice'].astype(int)
# create a list of features
feature_cols = ['MSZoning_num',          
'Street_num',             
'Utilities_num',          
'Neighborhood_num',       
'HouseStyle_num',         
'RoofStyle_num',          
'ExterCond_num',          
'Foundation_num',         
'BsmtQual_num',           
'BsmtCond_num',                    
'GarageQual_num',         
'GarageCond_num',         
'SaleType_num',           
'SaleCondition_num',      
'PoolArea',                             
'MoSold',                
'YrSold',                
'Fireplaces',            
'GarageArea',            
'FullBath',              
'HalfBath',              
'BedroomAbvGr',          
'TotRmsAbvGrd',          
'YearRemodAdd',          
'LotArea']
X = df[feature_cols]
y = df.SalePrice
# import class, instantiate estimator, fit with all data
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
rfclf.fit(df[feature_cols], df.SalePrice)
# compute the feature importances
a = pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})
model = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
model.fit(X, y)

feature_importance = model.feature_importances_
feature_importance = rfclf.feature_importances_
features = feature_cols
plt.figure(figsize=(16, 6))
plt.yscale('log', nonposy='clip')

plt.bar(range(len(feature_importance)), feature_importance, align='center')
plt.xticks(range(len(feature_importance)), features, rotation='vertical')
plt.title('Feature importance')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()
from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X, y) # FIT
predicted = clf.predict(X) # PREDICT
plt.scatter(y, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($00000s)')
plt.ylabel('Predicted price ($100000s)')
from sklearn.tree import DecisionTreeRegressor
# Instantiate the model, fit the results, and scatter in vs. out

clf = DecisionTreeRegressor()

clf.fit(X, y) # FIT
predicted = clf.predict(X) # PREDICT
plt.scatter(y, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($00000s)')
plt.ylabel('Predicted price ($100000s)')
## Predicting Home Prices: a Simple Linear Regression (By Using Train Data)
X = train[feature_cols]
y = train.SalePrice
from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X, y) # FIT
predicted = clf.predict(X) # PREDICT
plt.scatter(y, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($00000s)')
plt.ylabel('Predicted price ($100000s)')
from sklearn.tree import DecisionTreeRegressor
# Instantiate the model, fit the results, and scatter in vs. out

clf = DecisionTreeRegressor()

a = clf.fit(X, y) # FIT
predicted = clf.predict(X) # PREDICT
plt.scatter(y, predicted)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($00000s)')
plt.ylabel('Predicted price ($100000s)')

### SCIKIT-LEARN ###

X = train[feature_cols]
y = train.SalePrice

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print('Intercept',lm2.intercept_)
#print(lm2.coef_)
k=lm2.coef_.mean()
print('Coefficient:',k)
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
X = train[feature_cols]
y = train.SalePrice
# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
# Determining the property that are Prime Property  with 2 and average property with 1
train['price_condition'] = np.nan
train['price_condition'].loc[train["SalePrice"] <= 250000] = '1'
train['price_condition'].loc[train["SalePrice"] > 250000] = '2'
train['price_condition'].isnull().sum()
train['price_condition'].unique()
### SCIKIT-LEARN ###

# create X and y
feature_cols = ['SalePrice']
X = train[feature_cols]
y = train.price_condition

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print (lm2.intercept_)
print (lm2.coef_)
# manually calculate the prediction
0.526434394733148 + 3.43904283e-06*500000
### STATSMODELS ###

# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'SalePrice': [500000]})

# predict for a new observation
lm2.predict(X_new)
f, ax = plt.subplots(1,2, figsize=(20,10))

colors = ["#3791D7", "#D72626"]
labels ="Prime Properties", "Average Properties"

plt.suptitle('Classification of Dwelling by SalePrice ', fontsize=20)

train["price_condition"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)

#ax[0].set_ylabel('% of Condition of Dwelling', fontsize=14)

palette = ["#ddff33", "#3791D7"]

sns.barplot(x="YrSold", y="SalePrice", hue="price_condition", data=train, palette=palette, estimator=lambda x: len(x) / len(train) * 100)
ax[1].set(ylabel="(%)")
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
### STATSMODELS ###

# create a fitted model
lm1 = smf.ols(formula='SalePrice ~ HouseStyle_num', data=train).fit()

# print the coefficients
lm1.params
### SCIKIT-LEARN ###

# create X and y
feature_cols = ['HouseStyle_num']
X = train[feature_cols]
y = train.SalePrice

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print (lm2.intercept_)
print (lm2.coef_)
# manually calculate the prediction
# Here 7 representing SLvl	Split Level
158168.84 + 7488.37*7
### STATSMODELS ###

# print the confidence intervals for the model coefficients
lm1.conf_int()
### STATSMODELS ###

# print the p-values for the model coefficients
lm1.pvalues
### STATSMODELS ###

# print the R-squared value for the model
lm1.rsquared
### SCIKIT-LEARN ###

# print the R-squared value for the model
lm2.score(X, y)
### STATSMODELS ###

# create a fitted model with all three features
lm1 = smf.ols(formula='SalePrice ~ HouseStyle_num + LotArea + Neighborhood_num', data=train).fit()

# print the coefficients
lm1.params
### SCIKIT-LEARN ###

# create X and y
feature_cols = ['HouseStyle_num', 'LotArea', 'Neighborhood_num']
X = train[feature_cols]
y = train.SalePrice

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)

# print the coefficients
print ('Intercept:',lm2.intercept_)
print ('--------------------------')
print ('Cofficient:',(lm2.coef_))
### STATSMODELS ###

# print a summary of the fitted model
lm1.summary()

