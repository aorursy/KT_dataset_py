import pandas as pd

import numpy as np

import seaborn as sns
trainData=pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip")
df=pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip")
column_names = ["id","Prediction"]

sample=pd.DataFrame(columns=column_names)

sample["id"]=df["Id"]
trainData
trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')

df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y')
trainData['OpenDays']=""

df['OpenDays']=""
dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(trainData)]) })

dateLastTest = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(df)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y') 

dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%m/%d/%Y') 
df.head()
trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']

df['OpenDays'] = dateLastTest['Date'] - df['Open Date']
trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int)

df['OpenDays'] = df['OpenDays'].astype('timedelta64[D]').astype(int)
cityPerc = trainData[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()
sns.barplot(x='City Group', y='revenue', data=cityPerc)
citygroupDummy = pd.get_dummies(trainData['City Group'])

citygroupD = pd.get_dummies(df['City Group'])

citygroupDummy
trainData = trainData.join(citygroupDummy)

df = df.join(citygroupD)
trainData = trainData.drop('City Group', axis=1)

df = df.drop('City Group', axis=1)
trainData = trainData.drop('Open Date', axis=1)

df = df.drop('Open Date', axis=1)
trainData[["City","revenue"]].groupby(["City"]).mean().plot(kind="bar")
mean_revenue_per_city = trainData[['City', 'revenue']].groupby('City', as_index=False).mean()

mean_revenue_per_city['revenue'] = mean_revenue_per_city['revenue'].apply(lambda x: int(x/1e6)) 
mean_revenue_per_city
mean_dict = dict(zip(mean_revenue_per_city.City, mean_revenue_per_city.revenue))
mean_dict
trainData.replace({"City":mean_dict}, inplace=True)
trainData.City.unique()
trainData.City.mean()
df.City.unique()
df.replace({"City":mean_dict}, inplace=True)
#adding 4 as it was the mean in traindata column
df['City'] = df['City'].apply(lambda x: 4 if isinstance(x,str) else x)
trainData.Type.unique()
from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()

lr2=LabelEncoder()
trainData["Type"]=lr.fit_transform(trainData["Type"])

df["Type"]=lr2.fit_transform(df["Type"])
X = trainData.drop(['revenue', 'Id'],axis=1)
X
df = df.drop(['Id'],axis=1)
Y=trainData["revenue"]
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
from math import sqrt
cv = KFold(n_splits=10, shuffle=True, random_state=108)

model = LGBMRegressor(n_estimators=200, learning_rate=0.01, subsample=0.7, colsample_bytree=0.8)



scores = []

for train_idx, test_idx in cv.split(X):

    X_train = X.iloc[train_idx]

    X_val = X.iloc[test_idx]

    y_train = Y.iloc[train_idx]

    y_val = Y.iloc[test_idx]

    

    model.fit(X_train,y_train)

    preds = model.predict(X_val)

    

    rmse = sqrt(mean_squared_error(y_val, preds))

    print(rmse)

    scores.append(rmse)



print("\nMean score %d"%np.mean(scores))
predictions = model.predict(df)

predictions
sns.distplot(predictions, bins=20)
import eli5

from eli5.sklearn import PermutationImportance
X = trainData.drop(['revenue', 'Id'],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)



from xgboost import XGBRegressor

xgb=XGBRegressor()

xgb.fit(X_train,Y_train)
perm = PermutationImportance(xgb, random_state=1).fit(X_train,Y_train)

eli5.show_weights(perm, feature_names = X_train.columns.to_list())
trainData['P29_to_City_mean'] = trainData.groupby('City')['P29'].transform('mean')

trainData['P17_to_City_mean'] = trainData.groupby('City')['P17'].transform('mean')

trainData['P28_to_City_mean'] = trainData.groupby('City')['P28'].transform('mean')

trainData['P1_to_City_mean'] = trainData.groupby('City')['P1'].transform('mean')

trainData['P27_to_City_mean'] = trainData.groupby('City')['P27'].transform('mean')

trainData['P20_to_City_mean'] = trainData.groupby('City')['P20'].transform('mean')
df['P29_to_City_mean'] = df.groupby('City')['P29'].transform('mean')

df['P17_to_City_mean'] = df.groupby('City')['P17'].transform('mean')

df['P28_to_City_mean'] = df.groupby('City')['P28'].transform('mean')

df['P1_to_City_mean'] = df.groupby('City')['P1'].transform('mean')

df['P27_to_City_mean'] = df.groupby('City')['P27'].transform('mean')

df['P20_to_City_mean'] = df.groupby('City')['P20'].transform('mean')
X = trainData.drop(['revenue', 'Id'],axis=1)
X
cv = KFold(n_splits=10, shuffle=True, random_state=108)

model = LGBMRegressor(n_estimators=200, learning_rate=0.01, subsample=0.7, colsample_bytree=0.8)



scores = []

for train_idx, test_idx in cv.split(X):

    X_train = X.iloc[train_idx]

    X_val = X.iloc[test_idx]

    y_train = Y.iloc[train_idx]

    y_val = Y.iloc[test_idx]

    

    model.fit(X_train,y_train)

    preds = model.predict(X_val)

    

    rmse = sqrt(mean_squared_error(y_val, preds))

    print(rmse)

    scores.append(rmse)



print("\nMean score %d"%np.mean(scores))
predictions = model.predict(df)

sample['Prediction'] = predictions
sample
sample['Prediction']=sample['Prediction'].apply(lambda x: round((float(x/1e6)*1000000),1))
sample['Prediction']
sample.to_csv('submission.csv', index=False)