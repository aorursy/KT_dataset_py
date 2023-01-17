import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
filename = '/kaggle/input/bits-f464-l1/train.csv'
df = pd.read_csv(filename)
df.head(10)
df.isnull().any().any()
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

features=95

data=df.drop(columns=['label','id'])

sel = SelectKBest(f_regression, k=features)

sel.fit_transform(data,y=df['label'])

columns=sel.get_support(indices=True)

print(columns.size)

columns=data.iloc[:,columns].columns.to_list()

columns.insert(0, "id")

columns.append("label")

print(columns)

df=df[columns]

df.head()

print(df.shape)
split = True

par = 5

categorical_features = list()

numerical_features = list()

for column in df.drop(columns=['id','label']):

    uniques = df[column].unique().size

    if(df[column].unique().size == 1):

        df.drop(columns=[column])

        print("dropped %s" % column)

    elif split :

        if ((df[column]-df[column].astype(int))==0.0).all() and uniques <= par and uniques > 2:

            categorical_features.append(column)

            print("categorical %s unique vals = %d" % (column, df[column].unique().size))

        else:

            numerical_features.append(column)

if(not split):

    numerical_features = df.drop(columns=['label','id']).columns.to_list()

            

print(categorical_features)

print(numerical_features)
if(split):

    try:

        df=pd.get_dummies(data=df, columns=categorical_features)

    except KeyError:

        print("Maybe already hot encoded")

print(df.columns.size)

from sklearn.model_selection import train_test_split



X=df.drop(columns=['id','label'])

X_train,X_val,y_train,y_val = train_test_split(X,df['label'],test_size=0.05,random_state=42,shuffle=False)  



X.columns
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_val[numerical_features] = scaler.transform(X_val[numerical_features])



X_train[numerical_features].head()
skip=False

n_e=20

if not skip:

    from sklearn.linear_model import LinearRegression 

    from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import BaggingRegressor

    from sklearn.ensemble import RandomForestRegressor

    from sklearn.metrics import mean_squared_error

    regr1 = LinearRegression()

    regr2 = DecisionTreeRegressor()

    regr3 = RandomForestRegressor(n_estimators=n_e,

                                  max_features='auto',

                                  oob_score=True,

                                  n_jobs=-1,

                                  random_state=42,

                                  min_samples_leaf=2)

    regr4 = BaggingRegressor(regr3, 30, random_state=42)



    regr4.fit(X_train, y_train) 

    pred = regr4.predict(X_val)

    acc = mean_squared_error(pred, y_val)
if not skip:

    import os

    import time

    data = pd.DataFrame({"accuracy":[acc], "time":[time.ctime()], "splitting": split, "pars": par, "n_e":n_e})

    if not os.path.isfile('log.csv'):

        data.to_csv('log.csv', header=True, index=False)

    else:

        data.to_csv('log.csv', mode='a', header=False, index=False)

print(pd.read_csv("log.csv"))
save=False

if save:

    import pickle

    with open('model_%s'%n_e,'wb') as f:

        pickle.dump(regr4, f)
load=False

if load:

    from sklearn.metrics import mean_squared_error

    import pickle

    with open('model_%s'%n_e,'rb') as f:

        regr4 = pickle.load(f)

        pred = regr4.predict(X_val)

        acc = mean_squared_error(pred, y_val)

        print(acc)
checking = False

testfile = '/kaggle/input/bits-f464-l1/test.csv'

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error

tdf = pd.read_csv(testfile)

if checking:

    tdf = tdf.drop(columns=['label'])

columns=columns[:-1]

tdf=tdf[columns]

tdf=pd.get_dummies(tdf, columns=categorical_features)

print(tdf.columns.size)

X_final_test = tdf.drop(columns=['id'])

X_final_test[numerical_features] = scaler.transform(X_final_test[numerical_features])  

tpred = regr4.predict(X_final_test)

print(tpred)

ans = pd.DataFrame({'id':tdf['id'],'label':tpred})

ans.to_csv("submit.csv", index=False, header=True)

if checking:

    d = pd.read_csv("submit.csv")

    acc = mean_squared_error(d['label'], df['label'])

    print(acc)