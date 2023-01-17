#For Kaggle

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")
#Read Data

df_train=pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")
df_train.head()
print(f"Unique Countries: {len(df_train.Country_Region.unique())}")
train_dates=list(df_train.Date.unique())
print(f"Period : {len(df_train.Date.unique())} days")
print(f"From : {df_train.Date.min()} To : {df_train.Date.max()}")
print(f"Unique Regions: {df_train.shape[0]/len(df_train.Date.unique())}")
df_train.Country_Region.value_counts()
print(f"Number of rows without Country_Region : {df_train.Country_Region.isna().sum()}")
#New Column UniqueRegion combining Province_State and Country_Region
df_train["UniqueRegion"]=df_train.Country_Region
df_train.UniqueRegion[df_train.Province_State.isna()==False]=df_train.Province_State+" , "+df_train.Country_Region
df_train[df_train.Province_State.isna()==False]
df_train.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)
df_train
df_test.head()
test_dates=list(df_test.Date.unique())
print(f"Period :{len(df_test.Date.unique())} days")
print(f"From : {df_test.Date.min()} To : {df_test.Date.max()}")
print(f"Total Regions : {df_test.shape[0]/43}")
df_test["UniqueRegion"]=df_test.Country_Region
df_test.UniqueRegion[df_test.Province_State.isna()==False]=df_test.Province_State+" , "+df_test.Country_Region
df_test.drop(labels=["Province_State","Country_Region"], axis=1, inplace=True)
len(df_test.UniqueRegion.unique())
df_sub.head()
# Dates in train only
only_train_dates=set(train_dates)-set(test_dates)
print("Only train dates : ",len(only_train_dates))
#dates in train and test
intersection_dates=set(test_dates)&set(train_dates)
print("Intersection dates : ",len(intersection_dates))
#dates in only test
only_test_dates=set(test_dates)-set(train_dates)
print("Only Test dates : ",len(only_test_dates))
#Duplicate dataframe for test data with new column Delta
df_test_temp=pd.DataFrame()
df_test_temp["Date"]=df_test.Date
df_test_temp["ConfirmedCases"]=0.0
df_test_temp["Fatalities"]=0.0
df_test_temp["UniqueRegion"]=df_test.UniqueRegion
df_test_temp["Delta"]=1.0
%%time
import random
final_df=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion"])

for region in df_train.UniqueRegion.unique():
    df_temp=df_train[df_train.UniqueRegion==region].reset_index()
    df_temp["Delta"]=1.0
    size_train=df_temp.shape[0]
    for i in range(1,df_temp.shape[0]):
        if(df_temp.ConfirmedCases[i-1]>0):
            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]

    #number of days for delta trend
    n=4     

    #delta as average of previous n days
    delta_avg=df_temp.tail(n).Delta.mean()

    #delta as trend for previous n days
    delta_list=df_temp.tail(n).Delta

    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()

    df_test_app=df_test_temp[df_test_temp.UniqueRegion==region]
    df_test_app=df_test_app[df_test_app.Date>df_temp.Date.max()]

    X=np.arange(1,n+1).reshape(-1,1)
    Y=delta_list
    model=LinearRegression()
    model.fit(X,Y)

    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()

    for i in range (size_train, df_temp.shape[0]):
        n=n+1
        d=df_temp.Delta[i-1]*0.5+df_temp.Delta[i-5]*0.3*df_temp.Delta[i-10]*0.2
        m=model.predict(np.array([n]).reshape(-1,1))[0]
        choice=[m,d]
        df_temp.Delta[i]=max(1,random.choice(choice))
        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)


    size_test=df_temp.shape[0]-df_test_temp[df_test_temp.UniqueRegion==region].shape[0]

    df_temp=df_temp.iloc[size_test:,:]
    
    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","UniqueRegion"]]
    final_df=pd.concat([final_df,df_temp], ignore_index=True)
    
final_df.shape
df_sub.Fatalities=final_df.Fatalities
df_sub.ConfirmedCases=final_df.ConfirmedCases
df_sub.to_csv("submission.csv", index=None)
