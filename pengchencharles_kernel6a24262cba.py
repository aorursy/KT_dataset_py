# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OrdinalEncoder



#import matplotlib for visualization

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

df_sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")



print(df_train.shape)

print(df_test.shape)

print(df_sub.shape)

df_train.shape
df_train.head()
df_train.info()
df_train.describe()
df_train.Province_State.isna().sum()
len(df_train[df_train.Country_Region.isna()==False])
df_train.Country_Region.value_counts()
df_train[df_train.Province_State.isna()==False]
df_train['Region']=df_train['Country_Region']
df_train.Region[df_train.Province_State.isna()==False]=df_train['Province_State']+','+df_train['Country_Region']
df_train
len(df_train.Date.unique())
df_train.Date.min()
df_train.Date.max()
df_train.drop(labels=['Id','Province_State','Country_Region'], axis=1, inplace=True)
df_train
df_test
df_test.shape
df_test.info()
df_test.describe()
df_test.Date.min()
df_test.Date.max()
len(df_test.Date.unique())
df_test.shape[0]
df_test.Country_Region.value_counts()
df_test['Region']=df_test['Country_Region']
df_test.head()
df_test.Region[df_test.Province_State.isna()==False]=df_test.Province_State+','+df_test.Country_Region
df_test
len(df_test.Region.unique())
df_test.drop(labels=['ForecastId','Province_State','Country_Region'],axis=1,inplace=True)
len(df_test.Region.unique())
len(df_test.Date.unique())
df_sub
df_sub.info()
df_sub.describe()
df_sub.shape
train_dates=list(df_train.Date.unique())
test_dates=list(df_test.Date.unique())
only_train_dates=set(train_dates)-set(test_dates)
len(only_train_dates)
intersection_dates=set(train_dates)&set(test_dates)
len(intersection_dates)
only_test_dates=set(test_dates)-set(train_dates)
len(only_test_dates)
import random

df_test_temp=pd.DataFrame()

df_test_temp["Date"]=df_test.Date

df_test_temp["ConfirmedCases"]=0.0

df_test_temp["Fatalities"]=0.0

df_test_temp["Region"]=df_test.Region

df_test_temp["Delta"]=1.0
df_test_temp
%%time

final_df=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","Region"])
final_df
for region in df_train.Region.unique():

    df_temp=df_train[df_train.Region==region].reset_index()

    df_temp["Delta"]=1.0

    size_train=df_temp.shape[0]

    for i in range(1,df_temp.shape[0]):

        if(df_temp.ConfirmedCases[i-1]>0):

            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]


    #number of days for delta trend

    n=5     



    #delta as trend for previous n days

    delta_list=df_temp.tail(n).Delta

    

    #Average Growth Factor

    delta_avg=df_temp.tail(n).Delta.mean()



    #Morality rate as on last availabe date

    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()



    df_test_app=df_test_temp[df_test_temp.Region==region]

    df_test_app=df_test_app[df_test_app.Date>df_temp.Date.max()]



    X=np.arange(1,n+1).reshape(-1,1)

    Y=delta_list

    model=LinearRegression()

    model.fit(X,Y)

    #score_pred.append(model.score(X,Y))

    #reg_pred.append(region)



    df_temp=pd.concat([df_temp,df_test_app])

    df_temp=df_temp.reset_index()



    for i in range (size_train, df_temp.shape[0]):

        n=n+1        

        df_temp.Delta[i]=(df_temp.Delta[i-3]+max(1,model.predict(np.array([n]).reshape(-1,1))[0])+delta_avg)/3

        

    for i in range (size_train, df_temp.shape[0]):

        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)

        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)





    size_test=df_temp.shape[0]-df_test_temp[df_test_temp.Region==region].shape[0]



    df_temp=df_temp.iloc[size_test:,:]

    

    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","Region","Delta"]]

    final_df=pd.concat([final_df,df_temp], ignore_index=True)



#df_score=pd.DataFrame({"Region":reg_pred,"Score":score_pred})

#print(f"Average score (n={n}): {df_score.Score.mean()}")

#sns.distplot(df_score.Score)    

final_df.shape
df_sub.Fatalities=final_df.Fatalities

df_sub.ConfirmedCases=final_df.ConfirmedCases

df_sub.to_csv("submission.csv", index=None)