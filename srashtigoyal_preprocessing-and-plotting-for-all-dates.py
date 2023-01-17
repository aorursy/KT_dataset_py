import pandas as pd

#import pandas as pd

import numpy as np

#import seaborn as sns

from matplotlib import pyplot as plt

#import plotly.graph_objects as go

#from fbprophet import Prophet

import pycountry
df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

df.drop(columns=['Sno'],inplace=True)

#df_sort=df

#df_sort['Date']=pd.to_datetime(df['Date'].values)

#df_sort=df_sort.sort_values(["Date"])
df.head()
df.describe()
#df['Date'] =pd.to_datetime(df.Date)

df=df.groupby(['State/UnionTerritory',"Date"]).head()

States=np.unique(df['State/UnionTerritory'].values)

States
top5aff_states=df.groupby(['State/UnionTerritory']).max().sort_values(['ConfirmedIndianNational'],ascending=False)[:5].index.values
dates=df[df['State/UnionTerritory'] == 'Kerala']['Date'].values
for state in States:

    df1=df[df['State/UnionTerritory'] == state]

    

    rec_date_idx=np.where(dates==df1['Date'].values[0])[0][0]

    if rec_date_idx >0:

        df2=pd.DataFrame()

        df2['Date']=dates[:rec_date_idx]

        df2['ConfirmedIndianNational'] =  np.zeros(rec_date_idx)

        df2['ConfirmedForeignNational'] = np.zeros(rec_date_idx)

        df2['Cured']=np.zeros(rec_date_idx)

        df2['Deaths']=np.zeros(rec_date_idx)

        df2['State/UnionTerritory']=state

        df2=df2.append(df1,ignore_index=True)

    else: df2=df1

    df2.to_csv(state+'.csv',index=False)
plt.figure(figsize=(10,10))



for state in States:

    df1=pd.read_csv('/kaggle/working/'+state+'.csv')

    df1=df1[30:]

    plt.plot(df1['Date'],df1['ConfirmedIndianNational'],"*-",label=state)

    #np.savetxt(state+'.txt',df1['ConfirmedIndianNational']+df1['ConfirmedForeignNational'])

plt.xticks(rotation=90)

plt.legend()

plt.savefig('indian_states.png')
top5aff_states