import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

print(data.shape)

data.head(3)
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'])

print(test.shape)

test.head(3)
## Exploration all fileds. Especially, What fields include "NaN value"

data.info()
data['Dates-year'] = data['Dates'].dt.year

data['Dates-month'] = data['Dates'].dt.month

data['Dates-day'] = data['Dates'].dt.day

data['Dates-hour'] = data['Dates'].dt.hour

data['Dates-minute'] = data['Dates'].dt.minute

data['Dates-second'] = data['Dates'].dt.second
fig, ((axis1,axis2,axis3),(axis4,axis5,axis6)) = plt.subplots(nrows=2, ncols=3)

fig.set_size_inches(18,6)



sns.countplot(data=data, x='Dates-year', ax=axis1)

sns.countplot(data=data, x='Dates-month', ax=axis2)

sns.countplot(data=data, x='Dates-day', ax=axis3)

sns.countplot(data=data, x='Dates-hour', ax=axis4)

sns.countplot(data=data, x='Dates-minute', ax=axis5)

sns.countplot(data=data, x='Dates-second', ax=axis6)
fig, (axis1,axis2) = plt.subplots(nrows=2, ncols=1, figsize=(18,4)) 

sns.countplot(data=data, x='Dates-hour', ax=axis1)

sns.countplot(data=data, x='Dates-minute', ax=axis2)
# Dates-hour exploration

data['Dates-hour'].value_counts()[-5:]
## 



def bin_data_minute(hour):

    if hour >=8 & hour ==0:

        return 'High_hour'

    else:

        return 'Low_hour'
data['bin_dates_hour'] = data['Dates-hour'].apply(bin_data_minute)
fig, axis1 = plt.subplots(figsize=(10,20))

sns.countplot(data=data, y='Category', hue='bin_dates_hour',ax=axis1)
# Dates-minute exploration

data['Dates-minute'].value_counts()[:10]
# Number of addresses containing '/'

street_length = len(data[data['Address'].str.contains('/')])

print(street_length)



# Number of Block Addresses

print(len(data['Address'])- street_length)

def bin_address(address):

    if '/' in address:

        return 'Street'

    else:

        return 'Block'
data['Address_type'] = data['Address'].apply(bin_address)

data[['Address', 'Address_type']].head(5)
sns.countplot(data=data, x='Address_type')
# countplot에서 hue와 x / y의 차이를 두면 더 많은 시각화를 할 수 있다.

fig, axis1 = plt.subplots(figsize=(10,20))

sns.countplot(data=data, y='Category', hue='Address_type', ax=axis1)
# 아래의 결과 처럼 주소의 순서만 잘못 기입했지 같은 위치에서 일어난 범죄가 실재로 존재한다.

# 따라서 이 데이터들은 하나의 주소로 만들어 줘야한다.

# 탐색결과 'BLOCK' 주소는 순서가 바뀐 주소가 없었다.

print(len(data[data['Address'] == 'OAK ST / LAGUNA ST']))

print(len(data[data['Address'] == 'LAGUNA ST / OAK ST']))
# Street 주소의 unique한 값만 추출

crossload = data[data['Address'].str.contains('/')]['Address'].unique()

print('crossload의 개수 {0} 개'.format(len(crossload)))
# value_counts로 값을 세고 필요로 하는 값만 뽑아내는 방법 

# data[value_counts >= 100]을 하면 안된다.

# value_counts()의 반환 객체는 Series이고 >= 같은 연산자를 적용하면 Series객체의 Boolen값을 반환한다. 따라서 같은 Series객체로 감싸줘야한다.

# value_counts()는 unique()의 역할도 한다고 할 수 있다.

topN_address_list = data['Address'].value_counts()

topN_address_list = topN_address_list[topN_address_list >=100]

topN_address_list = topN_address_list.index

print('topN criminal address count is',len(topN_address_list))

# Modeling 을 위해 100개이하의 index들은 모두 MOdel에서 신경쓰지 않도록 'Others'로 선언

data['Address_clean'] = data['Address']

data.loc[~data['Address'].isin(topN_address_list), "Address_clean"] = 'Others'



data[['Address','Address_clean']].head(5)
crossload = data[data['Address_clean'].str.contains('/')]

print(crossload.shape)

crossload['Address_clean'].head(3)
crossload_list = crossload['Address_clean'].unique()

print('Before Adjustment ST_Address length is {0}' .format(len(crossload_list)))
from tqdm import tqdm
# 같은 종류의 street value를 어떻게 합칠지 확인하기



crossload_list[0].split('/')[1].strip() + " / " + crossload_list[0].split('/')[0].strip() 
# 같은 종류의 street value를 어떻게 합칠지 확인하기



crossload_list[0]
for address in tqdm(crossload_list):

    reverse_address = address.split('/')[1].strip() + " / " + address.split('/')[0].strip()

    data.loc[data['Address_clean'] == reverse_address, 'Address_clean'] = address

    
crossload_list = data[data['Address_clean'].str.contains('/')]

crossload_list = crossload_list['Address_clean'].unique()

print('Final ST_Address length is {0}' .format(len(crossload_list)))
data[['Category','PdDistrict']].head(3)
data['PdDistrict'].value_counts()
sns.countplot(data=data,  x='PdDistrict')
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

print(train.shape)

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'])

print(test.shape)
train['Dates-year'] = train['Dates'].dt.year

train['Dates-month'] = train['Dates'].dt.month

train['Dates-day'] = train['Dates'].dt.day

train['Dates-hour'] = train['Dates'].dt.hour

train['Dates-minute'] = train['Dates'].dt.minute

train['Dates-second'] = train['Dates'].dt.second



test['Dates-year'] = test['Dates'].dt.year

test['Dates-month'] = test['Dates'].dt.month

test['Dates-day'] = test['Dates'].dt.day

test['Dates-hour'] = test['Dates'].dt.hour

test['Dates-minute'] = test['Dates'].dt.minute

test['Dates-second'] = test['Dates'].dt.second

train.columns
print(train.shape)

train[['Dates-year', 'Dates-month',

       'Dates-day', 'Dates-hour', 'Dates-minute', 'Dates-second']].head(3)

print(test.shape)

test[['Dates-year', 'Dates-month',

       'Dates-day', 'Dates-hour', 'Dates-minute', 'Dates-second']].head(3)
train['Dates-minute_clean'] = train['Dates-minute']

test['Dates-minute_clean'] = test['Dates-minute']



train.loc[train['Dates-minute'] == 30, 'Dates-minute_clean'] = 0

train[train['Dates-minute'] == 30]



test.loc[test['Dates-minute'] == 30, 'Dates-minute_clean'] = 0

test[test['Dates-minute'] == 30]
fig, (axis1, axis2) = plt.subplots(2,1, figsize=(15,4))



sns.countplot(data=data, x='Dates-minute', ax=axis1)

sns.countplot(data=train, x='Dates-minute_clean', ax=axis2)
PdDistrict_dummies_train = pd.get_dummies(train['PdDistrict'], prefix='PdDistrict')

print(PdDistrict_dummies_train.shape)

PdDistrict_dummies_train.head(3)

PdDistrict_dummies_test = pd.get_dummies(test['PdDistrict'], prefix='PdDistrict')

print(PdDistrict_dummies_train.shape)

PdDistrict_dummies_train.head(3)
train2 = train.copy()

test2 = test.copy()
train = pd.concat([train2, PdDistrict_dummies_train], axis=1)

test = pd.concat([test2, PdDistrict_dummies_test], axis=1)



PdDistrict_columns_list = list(PdDistrict_dummies_train.columns)





print("The List of PdDistrict columns = {0}".format(PdDistrict_columns_list))



print(train.shape)

print(test.shape)



train[["PdDistrict"] + PdDistrict_columns_list].head()
DayOfWeek_dummies_train = pd.get_dummies(train['DayOfWeek'], prefix='DayOfWeek')

print(DayOfWeek_dummies_train.shape)

DayOfWeek_dummies_test = pd.get_dummies(test['DayOfWeek'], prefix='DayOfWeek')

print(DayOfWeek_dummies_test.shape)

DayOfWeek_dummies_train.head(3)
train2 = train.copy()

test2 = test.copy()
train = pd.concat([train2, DayOfWeek_dummies_train], axis=1)

test = pd.concat([test2, DayOfWeek_dummies_test], axis=1)



DayOfWeek_columns_list = list(DayOfWeek_dummies_train.columns)





print("The List of DayOfWeek columns = {0}".format(DayOfWeek_columns_list))



print(train.shape)

print(test.shape)



train[["DayOfWeek"] + DayOfWeek_columns_list].head()
train["Address_CrossRoad"] = train["Address"].str.contains("/")

test["Address_CrossRoad"] = test["Address"].str.contains("/")



print(train.shape)

print(test.shape)



train[["Address", "Address_CrossRoad"]].head()
major_address_list = train["Address"].value_counts()

major_address_list = major_address_list[major_address_list >= 100]

major_address_list = major_address_list.index



print("The number of major address = {0}".format(len(major_address_list)))

major_address_list[:5]
train["Address_clean"] = train["Address"]

test["Address_clean"] = test["Address"]



train.loc[~train["Address"].isin(major_address_list), "Address_clean"] = "Others"

test.loc[~test["Address"].isin(major_address_list), "Address_clean"] = "Others"



print(train.shape)

print(test.shape)



train[["Address", "Address_clean"]].head()
crossroad = train[train["Address_clean"].str.contains("/")]



print(crossroad.shape)

crossroad[["Address", "Address_clean", "Category"]].head()
crossroad_list = crossroad["Address_clean"].unique()



print("The number of cross road (Before) = {0}".format(len(crossroad_list)))

crossroad_list[:5]
from tqdm import tqdm



for address in tqdm(crossroad_list):

    address_split = address.split("/")

    reverse_address = address_split[1].strip() + " / " + address_split[0].strip()

    

    train.loc[train["Address_clean"] == reverse_address, "Address_clean"] = address

    test.loc[test["Address_clean"] == reverse_address, "Address_clean"] = address
print(len(train[train['Address_clean'] == 'JONES ST / TURK ST']))

print(len(train[train['Address_clean'] == 'TURK ST / JONES ST']))
feature_names = ['X', 'Y','Address_CrossRoad']

feature_names = feature_names + ['Dates-minute_clean','Dates-hour']

feature_names = feature_names + PdDistrict_columns_list

label_name = 'Category'

X_train = train[feature_names]

y_train = train[label_name]

print(X_train.shape)

print(y_train.shape)

X_train.head(3)
X_test = test[feature_names]

print(X_test.shape)

X_test.head(3)
# LabelEncoder convert Categorical variable to Numerical variable with 



from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()



label_encoder.fit(train["Address_clean"])

# fit을 한다는 것은 해당 Field를 normalize 하는 것이다.

# 예를들어, 중복되는 index들을 합쳐 기준을 세운다.



train["Address_clean_encode"] = label_encoder.transform(train["Address_clean"])

test["Address_clean_encode"] = label_encoder.transform(test["Address_clean"])

# 세운 기준을 통해 transform(숫자로 변경) 시킨다.





print(train.shape)

print(test.shape)



train[["Address", "Address_clean", "Address_clean_encode"]].head()
from sklearn.preprocessing import OneHotEncoder



one_hot_encoder = OneHotEncoder()



one_hot_encoder.fit(train[["Address_clean_encode"]])



train_address = one_hot_encoder.transform(train[["Address_clean_encode"]])

test_address = one_hot_encoder.transform(test[["Address_clean_encode"]])



print(train_address.shape)

print(test_address.shape)



train_address
# hstack 은 array를 병합해주는 함수이다. 이때 병렬적으로 연결해주기 때문에 빠른 연산이 가능하다.



from scipy.sparse import hstack



X_train = hstack((X_train.astype(np.float32), train_address))

# np 함수는 dataFrame에 모든 요소에 빠른 함수를 적용시킨다.



print(X_train.shape)

X_train
from scipy.sparse import hstack



X_test = hstack((X_test.astype(np.float32), test_address))



print(X_test.shape)

X_test
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score



seed = 37



model = xgb.XGBClassifier(objective='multi:softprob',

                          n_estimators=45,

                          learning_rate=1.0,

                          max_depth=6,

                          max_delta_step=1,

                          nthread=-1,

                          seed=seed)



%time score = cross_val_score(model, X_train, y_train, scoring='neg_log_loss', cv=5).mean()

score = score * -1.0



print("Score = {0:.5f}".format(score))
import pickle 



%time model.fit(X_train, y_train)



pickle.dump(model, open("models/xgboost.p", "wb"))
model = pickle.load(open("models/xgboost.p", "rb"))

model
predictions = model.predict_proba(X_test)



predictions = predictions.astype(np.float32)



print(predictions.shape)

predictions[:1]