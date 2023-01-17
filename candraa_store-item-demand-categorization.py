import pandas as pd 
import numpy as np
import os

df=pd.read_csv("fles.csv")
#Calculate the period of items used, and #times used from the df
df['ADI']=df['period']/['count']
#calculate coeficient of variation ,mean used,standard deviation of usage
df['CV2']=(df['std']/df['mean'])*(df['std']/df['mean'])

df['Flg1'] = "demand_profile"
df.loc[((df['ADI']<1.32) & (df['CV2']<0.49)), 'Flg1'] = "Smooth"
df.loc[((df['ADI']>=1.32) & (df['CV2']<0.49)), 'Flg1'] = "Intermittent"
df.loc[((df['ADI']<1.32) & (df['CV2']>=0.49)), 'Flg1'] = "Erratic"
df.loc[((df['ADI']>=1.32) & (df['CV2']>=0.49)), 'Flg1'] = "Lumpy"
group = pd.pivot_table(df, 
                       index='Flg1', 
                       #columns='domain', 
                       values='item', 
                       aggfunc='count', 
                       margins=True)