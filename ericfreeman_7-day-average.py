import numpy as np

import pandas as pd

from datetime import datetime

from datetime import timedelta

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

train_file = "../input/covid19-global-forecasting-week-5/train.csv"

test_file = "../input/covid19-global-forecasting-week-5/test.csv"

sub_file = "../input/covid19-global-forecasting-week-5/submission.csv"

Optimistic=True
df = pd.read_csv(train_file)

test_df =pd.read_csv(test_file)

sub_df =pd.read_csv(sub_file)

def preprocess(df):

    df["Date"] = df['Date'].astype("datetime64[ms]")

    df=df.sort_values(['Province_State','Country_Region','County','Date'])

    df['County'] = df['County'].fillna('None')

    df['Province_State'] = df['Province_State'].fillna('None')

    df['Country_Region'] = df['Country_Region'].fillna('None')

    return df
df = preprocess(df)

CC= df[df['Target']=="ConfirmedCases"]

FAT = df[df['Target']=="Fatalities"]



test_df =preprocess(pd.read_csv(test_file))

test_CC= test_df[test_df['Target']=="ConfirmedCases"]

test_FAT = test_df[test_df['Target']=="Fatalities"]
from datetime import timedelta

TEST_END = df["Date"].max()

TRAIN_LAST = datetime.strptime('20-04-27', '%y-%m-%d')

TRAIN_FIRST = df["Date"].max() - timedelta(days=5) #Use in cases where it's not clear when min date should be
def addstats(df):

    CCR = df.groupby(['County','Province_State','Country_Region']).TargetValue.rolling(7).agg(['mean','std']).reset_index()

    df = pd.merge(df,CCR[['level_3','mean','std']], left_index=True, right_on='level_3')

    return df



def merge_test(CC,test_CC):

    CC = CC.merge(test_CC,how='outer', on =['Province_State','Country_Region','County','Date'])

    CC= CC.sort_values(['Province_State','Country_Region','County','Date'])

    CC = CC.reset_index(drop=True)

    CC['mean'] = CC['mean'].fillna(method='ffill')

    CC['std'] = CC['std'].fillna(method='ffill')

    CC['7Delta'] = CC['mean'] - CC['mean'].shift(7)

    CC['mean'] = CC['mean'].fillna(method='ffill')

    CC['std'] = CC['std'].fillna(method='ffill')

    CC['7Delta'] = CC['7Delta'].fillna(method='ffill')

    return CC





def predict_target(CC):

    list1 =[]

    list2 =[]

    list3 = []

    pred = 0

    change = 0

    targetstd = 0

    pred_death = 0

    change_death = 0

    predict_days=1

    for index, row in CC.iterrows():

        if (row['Date']<=TRAIN_LAST) | (row['Date']==TRAIN_FIRST):

            change = row['7Delta']/7

            targetstd = row['std']

            pred = row['TargetValue']+change

            predict_days=1

        else:

            pred = pred+change

            predict_days=predict_days+1



        if row['Date']==TEST_END:

            change = row['7Delta']/7

            targetstd = row['std']

            pred = row['TargetValue']+change

            predict_days=1



        if row['Date']==TEST_END + timedelta(days=14):

            if Optimistic:

                if change >0:

                    change = -1*change/2

                else:

                    change/2



        pred = pred+change

        if pred<0:

            pred=0

        list2.append(pred)

        p05 = pred - 2*targetstd*predict_days/14

        if p05<0:

            p05=0

        p95 = pred + 2*targetstd*predict_days/14   

        list1.append(p05)

        list3.append(p95)

    CC['pred_CC_05']= list1

    CC['pred_CC_50']= list2

    CC['pred_CC_95']= list3

    return CC
CC = addstats(CC)

CC = merge_test(CC,test_CC)

CC = predict_target(CC)
FAT = addstats(FAT)

FAT = merge_test(FAT,test_FAT)

FAT = predict_target(FAT)
x="Hawaii"

#CC[(CC['Province_State']==x) & (CC.County=="None")]['TargetValue'].plot()

#CC[(CC['Province_State']==x) & (CC.County=="None")]['Predicted_CC_050'].plot()

CC[(CC['Province_State']==x) & (CC.County=="None")]['pred_CC_50'].plot()

CC[(CC['Province_State']==x) & (CC.County=="None")]['pred_CC_05'].plot()

CC[(CC['Province_State']==x) & (CC.County=="None")]['pred_CC_95'].plot()
x="New York"

FAT[(FAT['Province_State']==x) & (FAT.County=="None")]['TargetValue'].plot()

#CC[(CC['Province_State']==x) & (CC.County=="None")]['Predicted_CC_050'].plot()

FAT[(FAT['Province_State']==x) & (FAT.County=="None")]['pred_CC_50'].plot()

#FAT[(FAT['Province_State']==x) & (FAT.County=="None")]['pred_CC_05'].plot()

#FAT[(FAT['Province_State']==x) & (FAT.County=="None")]['pred_CC_95'].plot()
Both = CC.append(FAT)
Both = Both[~pd.isna(Both.ForecastId)]
Both_05 = Both[['ForecastId','pred_CC_05']]

Both_05['ForecastId'] = Both_05['ForecastId'].astype('int').astype(str) + "_0.05"

Both_05.columns = ['ForecastId_Quantile', 'TargetValue']

Both_50 = Both[['ForecastId','pred_CC_50']]

Both_50['ForecastId'] = Both_50['ForecastId'].astype('int').astype(str) + "_0.5"

Both_50.columns = ['ForecastId_Quantile', 'TargetValue']



Both_95 = Both[['ForecastId','pred_CC_95']]

Both_95['ForecastId'] = Both_95['ForecastId'].astype('int').astype(str) + "_0.95"

Both_95.columns = ['ForecastId_Quantile', 'TargetValue']
#Both_05
df_submit = pd.concat([Both_05, Both_50, Both_95])

df_submit = df_submit.sort_values('ForecastId_Quantile')
print(df_submit.shape)

df_submit.to_csv("submission.csv", index = False)

#df_submit.head(50)