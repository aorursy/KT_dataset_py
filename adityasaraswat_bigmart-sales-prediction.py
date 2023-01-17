# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/Train_UWu5bXk.txt')
train.head()
test=pd.read_csv('../input/Test_u94Q5KV.txt')
test.head()
train.info()
train.isnull().sum()
train.head(10)
train.columns
train['source']='train'

test['source']='test'
train.head()
data=pd.concat([train, test], ignore_index=True, sort=False)
data.tail()
train.shape, test.shape, data.shape
data.apply(lambda x: sum(x.isnull()))
data.describe()
data.nunique()
data.Item_Fat_Content.value_counts()
data.Item_Type.value_counts()
data.Outlet_Type.value_counts()
item_avg_weight=data.pivot_table(values='Item_Weight', index='Item_Identifier')
item_avg_weight
item_avg_weight.loc['DRA12']
data.Item_Identifier.shape
item_avg_weight.shape
miss_bool=data['Item_Weight'].isnull()
miss_bool
data.loc[miss_bool,'Item_Weight']=data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
from scipy.stats import mode
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
data.head()
outlet_size_mode
miss_bool_a=data['Outlet_Size'].isnull()
miss_bool_a
data.loc[miss_bool_a, 'Outlet_Size']=data.loc[miss_bool_a,'Outlet_Type'].apply(lambda x:outlet_size_mode[x])
data.isnull().sum()
data.isnull().sum()
data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')
visibility_avg=data.pivot_table(values='Item_Visibility', index='Item_Identifier')
visibility_avg
miss_bool_b=(data['Item_Visibility']==0)
miss_bool_b
data.loc[miss_bool_b,'Item_Visibility'] = data.loc[miss_bool_b,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
data.head(10)
data.isnull().sum()
data.Item_Visibility.isnull().sum()
data.head(10)
data['Item_Visibility_meanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
data.head(10)
data.nunique()
data['item_type_combined']=data['Item_Identifier'].apply(lambda x: x[0:2])
data.head()
data['item_type_combined']=data['item_type_combined'].map({'FD':'Food',

                                                             'NC':'Non-Consumable',

                                                             'DR':'Drinks'})
data['outlet_years']=2013-data['Outlet_Establishment_Year']
data.outlet_years.describe()
data['Item_Fat_Content'].value_counts()
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'LF':'Low Fat',

                                                             'reg':'Regular',

                                                             'low fat':'Low Fat'})
data.loc[data['item_type_combined']=='Non-Consumable', 'Item_Fat_Content']='Non-Edible'
data.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['outlet']=le.fit_transform(data['Outlet_Identifier'])
var_mod=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','item_type_combined','Outlet_Type','outlet']
le=LabelEncoder()

for i in var_mod:

    data[i]=le.fit_transform(data[i])

    
data.head()
data=pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',

                              'item_type_combined','outlet'])
data.head()
data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)
data.drop(['Item_Type','Outlet_Establishment_Year'], axis=1, inplace=True)
train_final=data.loc[data['source']=='train']

test_final=data.loc[data['source']=='test']
train_final.head()
test_final.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)

train_final.drop(['source'],axis=1,inplace=True)
train_final.info()
mean_sales=train_final['Item_Outlet_Sales'].mean()
base1=test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales']=mean_sales
base1.head()
base1.to_csv('alg0.csv', index=False)
#functioon for every model
target='Item_Outlet_Sales'

IDcol=['Item_Identifier','Outlet_Identifier']

#linear regression
import sklearn
from sklearn.linear_model import LinearRegression
alg1= LinearRegression()
predictors=[x for x in train_final.columns if x not in IDcol + [target]]
alg1.fit(train_final[predictors],train_final[target])
predictions=alg1.predict(test_final[predictors])
#submission
test_final[target]=predictions
submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('sol1.csv', index=False)
#COEF
pd.Series(alg1.coef_, predictors).sort_values().plot(kind='bar')
#RMSE
from sklearn import metrics

np.sqrt(metrics.mean_squared_error(train_final['Item_Outlet_Sales'], train_predictions))
#decision tree
from sklearn.tree import DecisionTreeRegressor
alg3=DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
alg3.fit(train_final[predictors],train_final['Item_Outlet_Sales'])
train_predictions=alg3.predict(train_final[predictors])
np.sqrt(metrics.mean_squared_error(train_final['Item_Outlet_Sales'],train_predictions))
pd.Series(alg3.feature_importances_,predictors).sort_values().plot(kind='bar')
predictions=alg3.predict(test_final[predictors])
test_final[target]=predictions
submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('sol2.csv', index=False)
test_final.head()
#reduce features
predictors = ['Item_MRP','Outlet_Type_0','outlet_5','outlet_years']
alg4=DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
alg4.fit(train_final[predictors],train_final['Item_Outlet_Sales'])
train_predictions=alg4.predict(train_final[predictors])
np.sqrt(metrics.mean_squared_error(train_final['Item_Outlet_Sales'],train_predictions))
predictions=alg4.predict(test_final[predictors])
test_final[target]=predictions
submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('sol4.csv',index=False)
#randomforest
from sklearn.ensemble import RandomForestRegressor
predictors=[x for x in train_final.columns if x not in IDcol + [target]]
alg5=RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
alg5.fit(train_final[predictors],train_final[target])
train_predictions=alg5.predict(train_final[predictors])
np.sqrt(metrics.mean_squared_error(train_final[target], train_predictions))
pd.Series(alg5.feature_importances_, predictors).sort_values().plot(kind='bar')
predictions=alg5.predict(test_final[predictors])
test_final[target]=predictions

submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('sol5.csv', index=False)
alg6=RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
alg6.fit(train_final[predictors],train_final[target])
train_predictions=alg6.predict(train_final[predictors])
np.sqrt(metrics.mean_squared_error(train_final[target], train_predictions))
pd.Series(alg6.feature_importances_, predictors).sort_values().plot(kind='bar')
predictions=alg6.predict(test_final[predictors])

test_final[target]=predictions

submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]

submission.to_csv('sol6.csv', index=False)
#reduce feature
predictors=[x for x in train_final.columns if x not in IDcol + [target]]
alg7=RandomForestRegressor(min_samples_split=50,max_depth=6, min_samples_leaf=50,max_features=5,n_estimators=400,n_jobs=4)
alg7.fit(train_final[predictors],train_final[target])
train_predictions=alg7.predict(train_final[predictors])
np.sqrt(metrics.mean_squared_error(train_final[target], train_predictions))
pd.Series(alg7.feature_importances_, predictors).sort_values().plot(kind='bar')
predictions=alg7.predict(test_final[predictors])
test_final[target]=predictions
submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('sol12.csv', index=False)
pd.Series(alg7.feature_importances_, predictors).sort_values()
#GBM
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4
predictors=[x for x in train_final.columns if x not in IDcol + [target]]
gbm0=GradientBoostingRegressor(random_state=10)
gbm0.fit(train_final[predictors],train_final[target])
train_predictions=gbm0.predict(train_final[predictors])
np.sqrt(metrics.mean_squared_error(train_final[target], train_predictions))
predictions=gbm0.predict(test_final[predictors])
test_final[target]=predictions
submission=test_final[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
submission.to_csv('sol13.csv', index=False)