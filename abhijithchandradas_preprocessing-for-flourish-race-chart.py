#Import Libraries



import os

import  pandas as pd

import numpy as np

from IPython.core.display import HTML



from warnings import filterwarnings

filterwarnings('ignore')
# Read Data

df=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

print(df.shape)

df.head()
df.drop(["Time", "ConfirmedIndianNational", "ConfirmedForeignNational"], axis=1, inplace=True)

df.rename(columns={"State/UnionTerritory":"State","Cured":"Rec","Deaths":"Death","Confirmed":"Conf"}, inplace=True)

df.Date=pd.to_datetime(df.Date, format="%d/%m/%y")

df["Act"]=df.Conf-df.Rec-df.Death

df["MortRate"]=round(df.Death*100/df.Conf,2)

df["RecRate"]=round(df.Rec*100/df.Conf,2)
date_list=df.Date.unique()

state_list=df.State.unique()

df_ts=pd.DataFrame({"State":state_list})

for date in date_list:

    df_ts[str(pd.to_datetime(date).date())]=0

    

for state in state_list:

    for date in date_list:

        df_ts[str(pd.to_datetime(date).date())][df_ts.State==state]=df.Conf[df.State==state][df.Date==date].sum()

df_ts.to_csv("Flourish_India_08_09.csv", index=None)