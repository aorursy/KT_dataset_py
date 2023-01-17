# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from IPython.display import display
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
test_file_path = '../input/test.csv' 
data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
#display(data.corr(method = 'pearson'))
cor_data = data.corr(method = 'pearson')   #Correlation Data
cor_in = cor_data.index[cor_data.SalePrice < 0.3].tolist()   ##find the indexes for low correlations
#display(cor_in)
data1 = data.drop(labels = cor_in, axis = 1)   #drop the columns with low corelation
test_data1 = test_data.drop(labels = cor_in, axis = 'columns')   #drop the columns in test data too
#display(data1)
data2 = data1.drop(labels = ['SalePrice'], axis = 'columns')
#c = list(data2.columns.values)
#selected_columns1 = ['LotArea', 'YearBuilt','Neighborhood','Condition1', '1stFlrSF', '2ndFlrSF', 'FullBath','GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'OverallCond', 'YearRemodAdd','BsmtFinSF1', 'GrLivArea', 'GarageCars']
#selected_columns = ['LotArea', 'YearBuilt','Neighborhood','Condition1', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageArea', 'OverallCond', 'YearRemodAdd','BsmtFinSF1', 'GarageCars']
#x = data[selected_columns]
#y = x.Neighborhood.value_counts()
#z = list(y.iloc[:])
#t = list(y.index)
#print(z)
#plt.bar(t,z)
#print(t)
#s = pd.isna(x).sum()
#print(s)
#df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],'C': [1, 2, 3]})
#print(pd.get_dummies(df))
#test_x = test_data1.drop(['Id'], axis = 'columns')
#print(data2.Condition2.unique())
#print(test_x.Condition2.unique())
#c1 = list(test_x.columns.values)
#print(len(c), len(c1))
data2 = data2.fillna(0)
test_x = test_data1.fillna(0)
data2 = pd.get_dummies(data2)
#print(data2.columns.values)
test_x = pd.get_dummies(test_x)
#print(len(data2.columns.values), len(test_x.columns.values))
datacol = data2.columns
testcol = test_x.columns
diff_col = datacol.difference(testcol)
diff_col2 = testcol.difference(datacol)
#diff_col = (set(data2.columns.values) ^ set(test_x.columns.values))
#print(diff_col)
t, u =test_x.shape
zerro = np.asarray([0]*t)
#print(set(data2.columns.values) ^ set(test_x.columns.values))
for i in diff_col:
    test_x[i]= zerro
#diff_col2 = set(data2.columns.values) ^ set(test_x.columns.values)
t, u =data2.shape
zerro = np.asarray([0]*t)
for i in diff_col2:
    data2[i]= zerro
#print('here:')
#print(set(data2.columns.values) ^ set(test_x.columns.values))
#print(len(data2.columns.values), len(test_x.columns.values))
#data2 = data2.drop(z, axis = 'columns')
#print(set(data2.columns.values) ^ set(test_x.columns.values))
#print(data.columns.values=='Functional_0')
#print(data2.columns[data2.isna().sum()>0])
my_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
data2 = pd.DataFrame(my_imputer.fit_transform(data2))
test_x = pd.DataFrame(my_imputer.fit_transform(test_x))
#print(len(data2.columns.values), len(test_x.columns.values))
#print(data2.shape)
#print(test_x.shape)
#print(test_x.head())
#print(data2.head())
y = data1.SalePrice
#price_model = RandomForestRegressor(1000)
price_model = XGBRegressor(n_estimators=1000, learning_rate = 0.009)
price_model.fit(data2, y)
predicted_prices = price_model.predict(test_x)
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
print(my_submission)
my_submission.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.