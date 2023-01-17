import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

test= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col=0)
train.head(2)
test.head(2)
train.info()
#去除训练集缺失过多的特征

Train = train.dropna(thresh=1000,axis=1)



#去除重复值

Train = Train.drop_duplicates()

Train.info()
#处理测试集

Test = test[Train.columns[:-1]]

Test.info()
#利用标签编码将训练集和测试集中的object对象转化为数值类型,

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



feature_all = pd.concat([Train.drop('SalePrice',axis=1),Test])

for i in feature_all.columns:

    #缺失值用众数填充

    feature_all[i].fillna(np.argmax(feature_all[i].value_counts()),inplace=True)

    if feature_all[i].dtype == 'object':

        feature_all[i] = pd.Series(label_encoder.fit_transform(feature_all[i].astype(str)),index=feature_all.index)
x_train_all = feature_all.iloc[:Train.shape[0]]

x_test = Train[['SalePrice']]

y_train_all = feature_all.iloc[Train.shape[0]:]

Train.shape,x_train_all.shape,x_test.shape,Test.shape,y_train_all.shape
# from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_score

from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor()

rfr.fit(x_train_all,x_test)

rfr.score(x_train_all,x_test)
SalePrice = rfr.predict(y_train_all)

df = pd.DataFrame({'Id':test.index,'SalePrice':SalePrice})

df.to_csv('submission.csv',index=False)

df.head()