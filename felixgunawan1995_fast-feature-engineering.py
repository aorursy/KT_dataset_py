import pandas as pd
import numpy as np
# Load the data
price = pd.read_csv('../input/train.csv')
price.iloc[:10,:]
#check our dataset info, see if there's any incomplete column, and non numeric type 
#we will delete those incomplete (less than 50%) and convert non numerical to numerical data (one hot encoding) so ml model can process those
price.info()
#drop incomplete (useless) columns
price = price.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
price.describe()
#transform all categorical columns into numerical (one hot encoding) columns
price_mod = price
for i in price.columns:
    if(price[i].dtype == np.object):
        #transform non numerical column to numerical column (One Hot Encoding)
        ohe = pd.get_dummies(price[i])
        #rename weird columns
        cols = ohe.columns
        cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
        cols = cols.map(lambda x: x.replace('(', '') if isinstance(x, str) else x)
        cols = cols.map(lambda x: x.replace(')', '') if isinstance(x, str) else x)
        cols = cols.map(lambda x: x.join((i,x)) if isinstance(x, str) else x) #yeah i know, kind of messy if i do it this way, change it however you like lol
        ohe.columns = cols
        #add column to original dataframe
        price_mod = pd.concat([price_mod, ohe], axis=1, join='inner')
#save all columns, for use to process test set
all_column = price_mod.columns.get_values()
all_column = all_column.tolist()
#filter, use only useful types
price_mod = price_mod.select_dtypes(['float64','int64','uint8'])

# Create correlation matrix
corr_matrix = price_mod.corr().abs()

# sort and save correlation matrix
corr_price = pd.DataFrame(data=corr_matrix["SalePrice"].sort_values(ascending=False))

# drop column with low correlation matrix, use only 38 features (root of 1460)
# according to this paper(https://academic.oup.com/bioinformatics/article/21/8/1509/249540)
# and prepare X_train and y_train too
to_drop = corr_price.iloc[38:,:0].index.values.tolist()
price_mod = price_mod.drop(to_drop,axis=1)
y_train = price_mod['SalePrice']
price_mod = price_mod.drop(['SalePrice'],axis=1)
#fill NaN values
#price_mod.fillna(0, inplace=True)
price_mod = price_mod.transform(lambda price_mod: price_mod.fillna(price_mod.mean()))
X_train = price_mod
X_train.head()
#normalization
from sklearn import preprocessing
X_train = preprocessing.normalize(X_train)
#process test set
pricetest = pd.read_csv('../input/test.csv')
pricetest = pricetest.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
#save id column (for merge/join later on)
id_column = pricetest['Id']
#transform all categorical columns into numerical (one hot encoding) columns
pricetest_mod = pricetest
for i in pricetest.columns:
    if(pricetest[i].dtype == np.object):
        #transform non numerical column to numerical column (One Hot Encoding)
        ohe = pd.get_dummies(pricetest[i])
        #rename weird columns
        cols = ohe.columns
        cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
        cols = cols.map(lambda x: x.replace('(', '') if isinstance(x, str) else x)
        cols = cols.map(lambda x: x.replace(')', '') if isinstance(x, str) else x)
        cols = cols.map(lambda x: x.join((i,x)) if isinstance(x, str) else x)
        ohe.columns = cols
        #add column to original dataframe
        pricetest_mod = pd.concat([pricetest_mod, ohe], axis=1, join='inner')
#making sure it has all column (Same as training set), it might be different if test set do not contain all categorical values
for i in all_column:
    if i not in pricetest_mod:
        pricetest_mod[i] = 0
#filter, use only useful types
pricetest_mod = pricetest_mod.select_dtypes(['float64','int64','uint8'])
#pricetest_mod['Exterior2ndOtherOther'] = 0
pricetest_mod = pricetest_mod.drop(to_drop,axis=1)
pricetest_mod = pricetest_mod.drop(['SalePrice'],axis=1)
pricetest_mod.fillna(0, inplace=True)
X_test = pricetest_mod
X_test = preprocessing.normalize(X_test)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
#other models
#from sklearn import linear_model
#reg = linear_model.Ridge (alpha = .5)
#reg.fit(X_train, y_train)
#predict sale price
predictions = grid_search.predict(X_test)
#convert to pandas dataframe
predictions = pd.DataFrame(data=predictions,columns=['SalePrice'])
#join id column with prediction
predictions = pd.concat([id_column, predictions], axis=1, join='inner')
predictions.head()
#save results to csv
predictions.to_csv('predictions.csv', sep=',', index=False)