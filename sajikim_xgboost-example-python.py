import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
def dummy_date(df):

    df["year"] = df["Date"].apply(lambda x: x.split("-")[0])

    df["month"] = df["Date"].apply(lambda x: x.split("-")[1])

    df["day"] = df["Date"].apply(lambda x: x.split("-")[2])

    #df.drop("Date",inplace=True,axis=1)

    return df



def LabelEncord_categorical(df):

    categorical_params = ["year","month","day"]

    for params in categorical_params:

        le = LabelEncoder()

        df[params] = le.fit_transform(df[params])

    return df



def dummies(df):

    categorical_params = ["year","month","day"]

    for params in categorical_params:

        dummies =  pd.get_dummies(df[params])

        df = pd.concat([df, dummies],axis=1)

    return df



def pre_processing(df):

    df = dummy_date(df)

    df = LabelEncord_categorical(df)

    df = dummies(df)

    return df



# read dataset

df_cov = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv', header=0)



# pre processing

df_cov = pre_processing(df_cov)



# add index

df_cov["DateDummy"]=pd.to_datetime(df_cov["Date"])

df_cov.set_index('DateDummy', inplace=True)



# make diff series

df_cov["Price"] = df_cov["Price"].interpolate('time',axis=0,limit_direction='both')

df_cov["Price_diff"] = df_cov["Price"].diff()

df_cov["Price_diff"] = df_cov["Price_diff"].fillna(0)



# split dataset to train & test

train = df_cov['2019-12-31':'2020-04-28']

test = df_cov['2020-04-29':'2020-06-08']
y_train = train["Price_diff"].values

X_train = train.drop(["Price","Price_diff","Date"],axis=1).values

y_test = test["Price_diff"].values

X_test = test.drop(["Price","Price_diff","Date"],axis=1).values
gbm = xgb.XGBRegressor()

reg_cv = GridSearchCV(gbm, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]

                            ,'max_depth': [3,4,6], 'n_estimators': list(range(20, 101, 10))}, verbose=1)

reg_cv.fit(X_train,y_train)
reg_cv.best_params_
gbm = xgb.XGBRegressor(**reg_cv.best_params_)

gbm.fit(X_train,y_train)
predictions = gbm.predict(X_test)

predictions
gbm.score(X_train,y_train)
submission = pd.DataFrame({ 'Date': test['Date'],

                            'Price': predictions })

base = train.tail(1)["Price"]

for index, row in submission.iterrows():

    base = base + row["Price"]

    submission.at[index, 'Price'] = base

    

submission.tail()
submission.to_csv("/kaggle/working/submission.csv", index=False)
%matplotlib inline

import matplotlib.pyplot as plt



def show_price_chart(df1,df2):

    plt.figure(figsize=(10,6))

    plt.style.use('ggplot')

 

    plt.plot([x for x in df1.index], df1["Price"],lw=1, color="black",label="Actual values")

    plt.plot([x for x in df2.index], df2["Price"],lw=1, color="red",label="Xgboost prediction")

    plt.legend(loc='best')

    plt.title('Actual and Predicted values')

    plt.xlabel('Date')

    plt.ylabel('Price')

    xmin = df1.index[0]

    xmax = df1.index[-1]

    

    plt.ylim(0, 80)

    plt.xlim(xmin, xmax)

    plt.show()
df_plt = pd.DataFrame({ 'Date': test['Date'],

                        'Price': submission['Price'] })



df1 = pd.concat([train, test])

show_price_chart(df1,submission)