# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 필요한 라이브러리 import

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import sin, cos, sqrt, atan2, radians
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
from IPython.display import HTML
import base64

# 노트북 안에 그래프를 그리기 위해
%matplotlib inline

# 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일을 사용
plt.style.use('ggplot')

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
from subprocess import check_output

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None)

baseDir="../input"
print(check_output(["ls", baseDir]).decode("utf8")) #check the files available in the directory
train = pd.read_csv(baseDir+"/train_with_school_and_subway.csv")
test = pd.read_csv(baseDir+"/test_with_school_and_subway.csv")

train.shape
train.info()

#train_apart = pd.read_csv(baseDir+"/train.csv")
#test_apart = pd.read_csv(baseDir+"/test.csv")

#train_apart.info()
'''
corrMatt = train[["transaction_year_month", "year_of_completion", "exclusive_use_area", "floor",  "total_parking_capacity_in_site",
                 "total_household_count_in_sites", "apartment_building_count_in_sites", "tallest_building_in_sites" ,"lowest_building_in_sites" ,"supply_area", "room_id", 
                  "total_household_count_of_area_type", "room_count", "bathroom_count" ,"nearest_subway_distance",
                 "nearest_school_distance","address_by_law_x","city","heat_type","heat_fuel","front_door_structure","nearest_subway_index","nearest_school_code",
                  "operation_type","foundation_date","subway_line_count","transaction_real_price"]]
corrMatt = corrMatt.corr()
#print(corrMatt)

mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots()
fig.set_size_inches(20,20)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
'''
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split # import 'train_test_split'
from sklearn.ensemble import RandomForestRegressor # import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error # import metrics from sklearn
from time import time

def rmsle(predicted_values, actual_values, convertExp=True):

    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)

    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    # 값이 0일 수도 있어서 로그를 취했을 때 마이너스 무한대가 될 수도 있기 때문에 1을 더해 줌
    # 로그를 씌워주는 것은 정규분포로 만들어주기 위해
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)
    
    return score

def output_submission(test_key, prediction, id_column, prediction_column):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df["key"] = test_key
    print('Output complete')
    return df[[id_column, prediction_column]]
    
    
def output_submission2(train_key, prediction, train_org_price, prediction_column):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df["key"] = train_key
    df['org_real'] = train_org_price
    
    return df[[id_column, prediction_column,'org_real']]

def my_score(y_train, y_result):
    score = sqrt(mean_squared_error(y_train, y_result))
    return score


def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)



label_name = "transaction_real_price"

y_train = train[label_name].copy()
train.drop(label_name,axis=1, inplace=True)
test.drop(label_name,axis=1, inplace=True)
train_id = train['key'].copy()
test_id = test['key'].copy()

train.drop("key",axis=1, inplace=True)
test.drop("key",axis=1, inplace=True)

#train.drop("nearest_school_code",axis=1, inplace=True)
#test.drop("nearest_school_code",axis=1, inplace=True)

print(train_id.shape)
print(test_id.shape)
print(train.shape)
print(test.shape)
#drop_features = ["room_count","subway_line_count","foundation_date","operation_type","front_door_structure","heat_fuel","heat_type","nearest_school_code",""]
drop_features = ["front_door_structure","heat_fuel","heat_type"]

#for var in drop_features:
train.drop(drop_features, axis=1, inplace=True)
test.drop(drop_features, axis=1, inplace=True)

print(train.shape)
print(test.shape)
train.info()
feature_names = ["transaction_year_month", "exclusive_use_area", "floor",  "total_parking_capacity_in_site",
                 "total_household_count_in_sites", "apartment_building_count_in_sites", "tallest_building_in_sites" ,"lowest_building_in_sites" ,"supply_area",
                  "room_count", "bathroom_count","nearest_subway_distance", "nearest_school_distance"]

feature_names

X_train = train[feature_names]
print(X_train.shape)
X_train.head()

X_test = test[feature_names]
print(X_test.shape)
X_test.head()
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
categorical_feature_names = ["address_by_law", "nearest_school_index", "city","apartment_id","room_id","year_of_completion"]
#nearest_subway_index drop
for var in categorical_feature_names:
    X_train[var] = train[var].astype("category")
    X_test[var] = test[var].astype("category")
#categorical_feature_names= ["address_by_law", "nearest_school_index", "city","apartment_id"]
# label encoder
from sklearn.preprocessing import LabelEncoder

cols = categorical_feature_names
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(X_train[c].values)) 
    X_train[c] = lbl.transform(list(X_train[c].values))

# shape        
print('Shape train: {}'.format(X_train.shape))

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(X_test[c].values)) 
    X_test[c] = lbl.transform(list(X_test[c].values))

# shape        
print('Shape test: {}'.format(X_test.shape))
# temp for var in drop_features:
#drop_features = ["nearest_subway_index"]
#X_train.drop(drop_features, axis=1, inplace=True)
#X_test.drop(drop_features, axis=1, inplace=True)

#X_train['room_count'].replace(0, 1,inplace=True)
#X_test['room_count'].replace(0, 1,inplace=True)



print(X_test.isnull().sum())
X_test.info()
#model train

rfModel = RandomForestRegressor(n_estimators=100,verbose=1,n_jobs=-1,warm_start=True)
y_train_log = np.log1p(y_train)
#rfModel.fit(X_train, y_train_log)
#case step 1 graph

estimators = np.arange(10, 101, 10)
scores = []
for n in estimators:
    rfModel.set_params(n_estimators=n)
    rfModel.fit(X_train, y_train_log)
    scores.append(rfModel.score(X_train, y_train_log))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)

#case step 2 train only
#rfModel.fit(X_train, y_train_log)
# 피쳐 스코어 확인 하기.
feature_importances = pd.DataFrame(rfModel.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.head(50)
# train 스코어 확인 하기.
preds = rfModel.predict(X_train)
preds2 = np.exp(preds)

score = my_score(y_train,preds2)
print("score : {}".format(score))
prediction = rfModel.predict(X_test)

prediction = np.exp(prediction)
df = output_submission(test_id, prediction, 'key', 'transaction_real_price')
create_download_link(df)
#df.to_csv('mycsvfile.csv',index=False)
