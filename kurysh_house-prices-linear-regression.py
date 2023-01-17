import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
'''I use the most valuable 10 features for simple linear regression
    these features were examined using DecisionTree in advance'''

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#calculating some means for further feature engineering
df.BsmtQual.fillna(0, inplace=True)
a = df.groupby('BsmtQual')['SalePrice'].mean()

df.FireplaceQu.fillna('no', inplace=True)
fire = df.groupby('FireplaceQu')['SalePrice'].mean()

#data preprocessing and features engineering were united into f_eng function
def f_eng(df):

    #new features
    df['BsmtBath'] = df.BsmtFullBath + 0.5 * df.BsmtHalfBath
    df['Bath'] = df.FullBath + 0.5 * df.HalfBath
    
    df.BsmtQual.fillna(0, inplace=True)
    df['BsmtH'] = df.BsmtQual.apply(lambda x: a[x])
    
    df.FireplaceQu.fillna('no', inplace=True)
    df['FireQ'] = df.FireplaceQu.apply(lambda x: fire[x])
    
    #n/a replacement
    df.LotFrontage.fillna(df.LotFrontage.mean(), inplace=True)
    all_features = df.fillna(0)
        
    return  all_features[['GrLivArea', 'LotArea', 'GarageArea', 'YearBuilt', 'YearRemodAdd', 'Bath', 'BsmtBath', \
                          'BsmtH', 'LotFrontage', 'FireQ']]

X_train = f_eng(df)
y =  df.SalePrice.values.reshape(-1,1)

regressor = LinearRegression()  
regressor.fit(X_train, y)

#loading test data
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
X_test = f_eng(test_data)

pred = regressor.predict(X_test)[:,0]
result = pd.DataFrame({'Id':test_data.Id, 'SalePrice':pred})
result.to_csv('iter_18.csv', index=False)