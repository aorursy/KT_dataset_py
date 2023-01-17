import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/trainandvalid/TrainAndValid.csv')

test = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/Test.csv')
train.head()
test.info()
train.info()
train.describe(include='all')
train['SalePrice'] = np.log(train.SalePrice)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

from sklearn.ensemble import RandomForestRegressor
def model_score(model, X_trn, y_trn, X_val, y_val):

    '''

    Returns the RMSLE Score for the given model

    '''

    model.fit(X_trn, y_trn)

    pred =model.predict(X_val)

    return np.sqrt(mse(pred, y_val))
model= RandomForestRegressor()

feature = ['YearMade']
X_zero = train[feature]

y_zero = train.SalePrice
X_trn, X_val, y_trn, y_val = train_test_split(X_zero, y_zero, test_size=0.3, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
model1 = LinearRegression()

model_score(model1, X_trn, y_trn, X_val, y_val)
model2 = XGBRegressor()

model_score(model2, X_trn, y_trn, X_val, y_val)
#for i in range(50, 500, 50):

#    model3 = XGBRegressor(n_estimators=200)

#    scr = model_score(model3, X_trn, y_trn, X_val, y_val)

#    print(i, '\t', scr)
train.datasource.unique()
features = ['YearMade', 'datasource']
X_one = train[features]

y_one = train.SalePrice
X_trn, X_val, y_trn, y_val = train_test_split(X_one, y_one, test_size=0.3, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
features = ['YearMade', 'datasource', 'state']
X_two = train[features]

y_two = train.SalePrice
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

X_two['state']=enc.fit_transform(X_two.state)
X_trn, X_val, y_trn, y_val = train_test_split(X_two, y_two, test_size=0.2, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
train['age']= train.saledate.str[-9:-5].astype(int) - train.YearMade

test['age'] = test.saledate.str[-9:-5].astype(int) - test.YearMade
features = ['YearMade', 'datasource', 'state', 'age']

X_three = train[features]

y_three = train.SalePrice
X_three['state']= enc.fit_transform(X_three.state)
X_trn, X_val, y_trn, y_val = train_test_split(X_three, y_three, test_size=0.2, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
features = ['YearMade', 'datasource', 'state', 'age', 'fiBaseModel']

X_four = train[features]

y_four = train.SalePrice
X_four['state']= enc.fit_transform(X_four.state)

X_four['fiBaseModel']= enc.fit_transform(X_four.fiBaseModel)
X_trn, X_val, y_trn, y_val = train_test_split(X_four, y_four, test_size=0.2, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
model.score(X_val, y_val)
features = ['YearMade', 'datasource', 'state', 'age', 'fiBaseModel', 'fiProductClassDesc' ]

X_five = train[features]

y_five = train.SalePrice
X_five['state']= enc.fit_transform(X_five.state)

X_five['fiBaseModel']= enc.fit_transform(X_five.fiBaseModel)

X_five['fiProductClassDesc']= enc.fit_transform(X_five.fiProductClassDesc)
X_trn, X_val, y_trn, y_val = train_test_split(X_five, y_five, test_size=0.2, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
features = ['YearMade', 'datasource', 'state', 'age', 'fiBaseModel', 'fiProductClassDesc' , 'fiModelDesc']

X_six = train[features]

y_six = train.SalePrice
X_test = test[features]
net_state=X_six.state

net_state= net_state.append(X_test.state, ignore_index=True)
net_state
enc_st = LabelEncoder()

enc_st.fit(net_state)
X_six['state']= enc_st.transform(X_six.state)
X_test['state'] = enc_st.transform(X_test.state)
net_pcd = X_six.fiProductClassDesc

net_pcd = net_pcd.append(X_test.fiProductClassDesc, ignore_index=True)

net_pcd
enc_pcd = LabelEncoder()

enc_pcd.fit(net_pcd)
X_six.fiProductClassDesc = enc_pcd.transform(X_six.fiProductClassDesc)
X_test['fiProductClassDesc'] = enc_pcd.transform(X_test.fiProductClassDesc )
net_bm = X_six.fiBaseModel

net_bm = net_bm.append(X_test.fiBaseModel, ignore_index=True)

net_bm
enc_bm = LabelEncoder()

enc_bm.fit(net_bm)
X_six.fiBaseModel = enc_bm.transform(X_six.fiBaseModel)
X_test.fiBaseModel = enc_bm.transform(X_test.fiBaseModel)
X_test
net_md = X_six.fiModelDesc

net_md = net_md.append(X_test.fiModelDesc, ignore_index=True)

net_md
enc_md = LabelEncoder()

enc_md.fit(net_md)
X_six['fiModelDesc'] = enc_md.transform(X_six.fiModelDesc)
X_test['fiModelDesc'] = enc_md.transform(X_test.fiModelDesc)
X_six
X_trn, X_val, y_trn, y_val = train_test_split(X_six, y_six, test_size=0.2, random_state=0)
model_score(model, X_trn, y_trn, X_val, y_val)
model.score(X_val, y_val)
#for i in range(10,250, 10):

#    model = RandomForestRegressor(n_estimators= i)

#    print(i, '\t',  model_score(model, X_trn, y_trn, X_val, y_val))
model = RandomForestRegressor(n_estimators=110, n_jobs= -1)

model_score(model, X_trn, y_trn, X_val, y_val)
model.score(X_val, y_val)
X_six.describe()
model = RandomForestRegressor(max_depth=30, min_samples_split=20, n_estimators=110, n_jobs= -1)

model_score(model, X_trn, y_trn, X_val, y_val)
model.score(X_val, y_val)
train.Engine_Horsepower.unique()
train.fiBaseModel
train.state.unique()
test.state.unique()
pred = model.predict(X_test)
output = pd.DataFrame({'Id': test.SalesID,

                       'SalePrice': np.exp(pred)})

output.to_csv('submission.csv', index=False)