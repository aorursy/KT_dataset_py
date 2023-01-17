import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 



from xgboost import  XGBRegressor
%matplotlib inline
train_data = pd.read_csv("../input/train.csv",low_memory= False)

test_data = pd.read_csv("../input/test.csv",low_memory= False)

store_data = pd.read_csv("../input/store.csv",low_memory= False)

test_copy = test_data
print("Shape of Train data :", train_data.shape)

print("Shape of Test data :", test_data.shape)

print("Shape of Store data :", store_data.shape)
train_data.head()
test_data.head()
store_data.head(100)
train_data.isnull().sum()
test_data.isnull().sum()
store_data.isnull().sum().sort_values(ascending = False)
store_data['Promo2SinceWeek'].unique()
train_data['Store'].unique()
train_data['DayOfWeek'].unique()
train_data['Open'].unique()
train_data['StateHoliday'].unique()
train_data['Promo'].unique()
train_data['Store'].unique()
store_data['CompetitionOpenSinceMonth'].unique()
print(sum(train_data["Open"] == 0))

print(sum(train_data["Open"] == 1))
print(sum(test_data["Open"] == 0))

print(sum(test_data["Open"] == 1))
print(sum(train_data["StateHoliday"] == 'a'))

print(sum(train_data["StateHoliday"] == 'b'))

print(sum(train_data["StateHoliday"] == 'c'))

print(sum(train_data["StateHoliday"] == 0))
plt.plot(train_data['DayOfWeek'],train_data['Customers'])
train_data[['Sales','Customers','Promo','SchoolHoliday']].corr(method='pearson')
train_data['Mon'] = train_data["Date"].apply(lambda x : int(x[5:7]))

train_data['Yr'] = train_data["Date"].apply(lambda x : int(x[:4]))

train_data["HolidayBin"] = train_data.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
test_data['Mon'] = test_data["Date"].apply(lambda x : int(x[5:7]))

test_data['Yr'] = test_data["Date"].apply(lambda x : int(x[:4]))

test_data["HolidayBin"] = test_data.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})
train_data = train_data.merge(store_data)

test_data =test_data.merge(store_data)
train_data.head()
test_data.head()
train_data.isnull().sum().sort_values(ascending= False)
test_data.isnull().sum().sort_values(ascending= False)
test_data[test_data['Open'].isnull()]
for i in train_data['Promo2SinceWeek'].unique() :

    print(i ,':', sum(train_data['Promo2SinceWeek'] == i ))

for i in train_data['CompetitionOpenSinceMonth'].unique() :

    print(i ,':', sum(train_data['CompetitionOpenSinceMonth'] == i ))
for i in train_data['Promo2SinceYear'].unique() :

    print(i ,':', sum(train_data['Promo2SinceYear'] == i ))
for i in train_data['CompetitionOpenSinceYear'].unique() :

    print(i ,':', sum(train_data['CompetitionOpenSinceYear'] == i ))
train_data = train_data.drop(['Customers', 'Store','Date','StateHoliday'],axis= 1 )

test_data = test_data.drop(['Date','StateHoliday','Store','Id'],axis= 1 )
train_data.head()
test_data.head()
sum(train_data['Open'] == 0)
train_data = train_data.drop(train_data[train_data['Open'] == 0].index.tolist())
sum(train_data['Open'] == 0)
train_data.shape
train_data[train_data['HolidayBin'].isnull()]
train_data['CompetitionOpenSinceMonth'] = train_data['CompetitionOpenSinceMonth'].fillna(9.0)

train_data['HolidayBin'] = train_data['HolidayBin'].fillna(0)

train_data['Promo2SinceWeek'] = train_data['Promo2SinceWeek'].fillna(40.0)

train_data['Promo2SinceYear'] = train_data['Promo2SinceYear'].fillna(2012.0)

train_data['CompetitionOpenSinceYear'] = train_data['CompetitionOpenSinceYear'].fillna(2012.0)

train_data['CompetitionDistance'] = train_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].mean())



train_data.isnull().sum().sort_values(ascending = False)
test_data['Open'] = test_data['Open'].fillna(1)

test_data['CompetitionOpenSinceMonth'] = test_data['CompetitionOpenSinceMonth'].fillna(9.0)

test_data['CompetitionDistance'] = test_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].mean())

test_data['CompetitionOpenSinceYear'] = test_data['CompetitionOpenSinceYear'].fillna(2012.0)

test_data['Promo2SinceWeek'] = test_data['Promo2SinceWeek'].fillna(40.0)

test_data['Promo2SinceYear'] = test_data['Promo2SinceYear'].fillna(2012.0)



test_data.isnull().sum().sort_values(ascending = False)
train_data.shape
sum(train_data['Sales'] < 0 )
test_data.shape
train_data.head(100)
categorical_train = train_data.columns.tolist()

print(categorical_train)

train_data[categorical_train].corr(method='pearson')
train_features = train_data.drop(['Open'],axis = 1)

categorical_train = train_features.columns.tolist()

print(categorical_train)

train_data[categorical_train].corr(method='pearson')

train_features = train_data.drop(['Sales'],axis = 1)

full_features = pd.concat([train_features,test_data],ignore_index= True)

print(train_features.shape)

print(test_data.shape)
full_features.head()
full_features.shape
full_features = pd.get_dummies(full_features,columns= ['HolidayBin','Assortment','StoreType'])
full_features.shape
full_features = full_features.drop('PromoInterval',axis = 1)
train_features = full_features.iloc[:844392,:].values

test_data = full_features.iloc[844392:,:].values

train_sales = train_data['Sales'].values
print(train_features.shape)

print(train_sales.shape)

print(test_data.shape)
#train_sales = np.log(train_sales)
xgboost = XGBRegressor(learning_rate=0.009, n_estimators=500,

                                     max_depth=10, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006, random_state=42)
xgboost.fit(train_features,train_sales)

predictions = xgboost.predict(test_data)
#preds = np.exp(predictions)
pred_df = pd.DataFrame({"Id": test_copy["Id"], 'Sales': predictions})

pred_df.to_csv("xgboost_4_submission.csv", index=False)