# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
# Input data files are available in the "../input/" directory.
#Input of Data 
train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)
train['dataset'] = "train"
test['dataset'] = "test"
data = pd.concat([train,test], axis = 0)
categorical = ['property_type','room_type','bed_type','cancellation_policy','city']
data = pd.get_dummies(data, columns = categorical)
data.amenities = data.amenities.str.replace("[{}]", "") 
amenity_one = data.amenities.str.get_dummies(sep = ",")
data = pd.concat([data,amenity_one], axis=1).drop('amenities', axis = 1)

data.host_has_profile_pic = data.host_has_profile_pic.replace("t", 1) 
data.host_has_profile_pic = data.host_has_profile_pic.replace("f", 0) 

data.host_identity_verified = data.host_identity_verified.replace("t", 1) 
data.host_identity_verified = data.host_identity_verified.replace("f", 0) 

data.instant_bookable = data.instant_bookable.replace("t", 1) 
data.instant_bookable = data.instant_bookable.replace("f", 0)

data.host_response_rate = data.host_response_rate.str.replace("%","") 
data.host_response_rate[data.host_response_rate.isnull()] =0
data['host_response_rate'] = data.host_response_rate.astype(object).astype(int)

data['cleaning_fee'] = data['cleaning_fee']*1
data['first_review']=pd.to_datetime(data['first_review'])
data['first_review_Year'] = data['first_review'].dt.year
data['first_review_Month'] = data['first_review'].dt.month
data['first_review_Day'] =data['first_review'].dt.day
                     
data['last_review']=pd.to_datetime(data['last_review'])
data['last_review_Year'] = data['last_review'].dt.year
data['last_review_Month'] = data['last_review'].dt.month
data['last_review_Day'] = data['last_review'].dt.day

data['host_since']=pd.to_datetime(data['host_since'])
data['host_since_Year'] = data['host_since'].dt.year
data['host_since_Month'] = data['host_since'].dt.month
data['host_since_Day'] = data['host_since'].dt.day

data['thumbnail_url'] = data['thumbnail_url'].where(data['thumbnail_url'].isnull(), 1).fillna(0).astype(int)
# Select only numeric data and impute missing values as 0
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_x = data[data.dataset == "train"].select_dtypes(include=numerics).drop("log_price", axis = 1).fillna(0) .values

test_x = data[data.dataset == "test"].select_dtypes(include=numerics).drop("log_price", axis = 1).fillna(0).values
    
train_y = data[data.dataset == "train"].log_price.values
gbm = xgb.XGBRegressor(objective="reg:linear",max_depth=5, n_estimators=50, learning_rate=0.5).fit(train_x, train_y,eval_metric='rmse', verbose = True)
#gbm.fit(train_x, train_y)
#final_prediction = gbm.predict(test_x)

#submission = pd.DataFrame(np.column_stack([test.id, final_prediction]), columns = ['id','log_price'])
#submission.to_csv("sample_submission_gbm1.csv", index = False)


print(gbm.feature_importances_)

from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(gbm)
pyplot.show()
