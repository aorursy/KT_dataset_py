# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# importing the data
def fetch_data(path,data_set):
    dest = os.path.join(path,data_set)
    return pd.read_csv(dest)
Path = "../input/"
train_data = fetch_data(Path,"train.csv")
test_data = fetch_data(Path,"test.csv")
train_data.head()

col = ['Alley','FireplaceQu','Fence','PoolQC','MiscFeature','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageFinish','GarageQual','GarageCond']
for col in col:
    train_data[col] = train_data[col].fillna('None')
train_data.head()
from sklearn.model_selection import train_test_split
def split_data(data,ratio):
    return train_test_split(data,test_size = ratio,random_state=42)

train_data_df,val_data = split_data(train_data,10) 
train_data_df.describe().T
train_data_df.describe(include = ['O']).T
train_data_df.dtypes[train_data_df.dtypes=='object'].index 



train_data_df.hist(bins = 10 ,figsize = (30,30))
plt.show()

from sklearn.preprocessing import(Imputer,MultiLabelBinarizer,StandardScaler,PolynomialFeatures)
from  sklearn.base import (BaseEstimator,TransformerMixin,clone)
from sklearn.pipeline import(Pipeline,FeatureUnion)
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
class MyLabelBinarizer(TransformerMixin):
    def __init__(self):
        self.encoder = MultiLabelBinarizer()
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

    
drop = ['Id','SalePrice']
housing_data_lables = train_data_df['SalePrice']
housing = train_data_df.drop(drop,axis = 1)
feature_char_new = list(housing.describe(include=['O']))
feature_num_new = list(housing.describe(exclude=['O']))
num_pipeline = Pipeline([('data_frame_selector',DataFrameSelector(feature_num_new)),('imputer',Imputer(strategy = 'median')),('std_scaler',StandardScaler())])
char_pipeline = Pipeline([('data_frame_selector',DataFrameSelector(feature_char_new)),('label_binarizer',MyLabelBinarizer())])
full_pipeline = FeatureUnion(transformer_list= [('num_pipeline',num_pipeline),('char_pipeline',char_pipeline)])
x_train, y_train = full_pipeline.fit_transform(housing), housing_data_lables




x_val,y_val = full_pipeline.transform(val_data),val_data['SalePrice']
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
gbrt = GradientBoostingRegressor(max_depth = 400, warm_start = True)
error = float("inf")
count = 0
for n_estimators in range(1,500) :
    gbrt.n_estimators = n_estimators
    gbrt.fit(x_train,y_train)
    predict = gbrt.predict(x_val)
    val_error = np.square(np.log(predict + 1) - np.log(y_val + 1)).mean() ** 0.5
    if val_error < error:
        error = val_error
        count = 0
    else:
        count = count+1
        if count == 10:
            best_model = clone(gbrt)
            break
            
best_model
predict = gbrt.predict(x_val)
np.square(np.log(predict + 1) - np.log(y_val + 1)).mean() ** 0.5
col = list(test_data.describe(include = ['O']))
for col in col:
  test_data[col] = test_data[col].fillna('None')
test_data_prepared = test_data.drop(['Id'],axis=1)
x_test = full_pipeline.transform(test_data_prepared)
test_predict = gbrt.predict(x_test)
test_predict
submission = pd.DataFrame(test_predict,columns = ['SalePrice'],index = test_data['Id'])
submission.to_csv("submission.csv")