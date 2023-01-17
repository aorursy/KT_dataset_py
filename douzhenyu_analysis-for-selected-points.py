# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from matplotlib import dates as md
import seaborn as sns
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)
ls
list = [1,2,3,4,5,9,11,12,13,25,26,27,29]
df_T_all=pd.DataFrame()
df_W_all=pd.DataFrame()
df_June_T_all=pd.DataFrame()
df_June_W_all=pd.DataFrame()
df_Nov_T_all=pd.DataFrame()
df_Nov_W_all=pd.DataFrame()
for i in list:
  #temp
  df_T=pd.read_csv('point'+str(i)+'.csv',usecols=['timestamp', 'Temp',], index_col = 'timestamp', parse_dates = True)
  df_T = df_T[(df_T['Temp'] > 0)]
  df_T_m= df_T.resample('M').mean()
  df_T_m.rename(columns={'Temp':'T-'+str(i)},inplace=True)
  df_T_all=pd.concat([df_T_all,df_T_m],axis=1)
  #wind
  df_W=pd.read_csv('point'+str(i)+'.csv',usecols=['timestamp', 'Wind Speed',], index_col = 'timestamp', parse_dates = True)
  df_W = df_W[(df_W['Wind Speed'] < 3)&(df_W['Wind Speed']>=0)]
  df_W_m= df_W.resample('M').mean()
  df_W_m.rename(columns={'Wind Speed':'WS-'+str(i)},inplace=True)
  df_W_all=pd.concat([df_W_all,df_W_m],axis=1)
  #concat June
  df_T_June=df_T.truncate(before='2016-06-01',after='2016-07-01').resample('1H').mean()
  df_T_June['hour']=df_T_June.index.hour
  df_T_June=df_T_June.groupby('hour').mean()
  df_T_June.rename(columns={'Temp':'T-'+str(i)},inplace=True)
  df_W_June=df_W.truncate(before='2016-06-01',after='2016-07-01').resample('1H').mean()
  df_W_June['hour']=df_W_June.index.hour
  df_W_June=df_W_June.groupby('hour').mean()
  df_W_June.rename(columns={'Wind Speed':'WS-'+str(i)},inplace=True)
  df_June_T_all=pd.concat([df_June_T_all,df_T_June],axis=1)
  df_June_W_all=pd.concat([df_June_W_all,df_W_June],axis=1)
  #concat Nov
  df_T_Nov=df_T.truncate(before='2016-11-01',after='2016-12-01').resample('1H').mean()
  df_T_Nov['hour']=df_T_Nov.index.hour
  df_T_Nov=df_T_Nov.groupby('hour').mean()
  df_T_Nov.rename(columns={'Temp':'T-'+str(i)},inplace=True)
  df_W_Nov=df_W.truncate(before='2016-11-01',after='2016-12-01').resample('1H').mean()
  df_W_Nov['hour']=df_W_Nov.index.hour
  df_W_Nov=df_W_Nov.groupby('hour').mean()
  df_W_Nov.rename(columns={'Wind Speed':'WS-'+str(i)},inplace=True)
  df_Nov_T_all=pd.concat([df_Nov_T_all,df_T_Nov],axis=1)
  df_Nov_W_all=pd.concat([df_Nov_W_all,df_W_Nov],axis=1)
df_T_all
df_W_all
df_T_all.plot(figsize=(20,10))
plt.title('Temp of 2016',size=20)
plt.xlabel('Month')
plt.ylabel('Temp,℃')
plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
Tmaxmin={'Type':['max','min'],'June':[7,0],'July':[2,0],'Nov':[0,11],'Feb':[0,1]}
df_Tmaxmin=pd.DataFrame(Tmaxmin)
df_Tmaxmin.set_index('Type',inplace=True)

df_Tmaxmin
df_Tmaxmin.plot.bar(figsize=(10,5))
plt.title('Tmax and Tmin',size=20)
plt.xlabel('Type')
plt.ylabel('amount')
plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
df_June_T_all
df_June_T_all.iplot(size=(20,5),title='Temp of 24h in June',xTitle='Time of Day', yTitle='Temp,℃')
df_June_W_all
df_June_W_all.iplot(size=(20,5),title='Wind Speed of 24h in June',xTitle='Time of Day', yTitle='Wind Speed,m/s')
df_Nov_T_all
df_Nov_T_all.iplot(size=(20,5),title='Temp of 24h in Nov',xTitle='Time of Day', yTitle='Temp,℃')
df_Nov_W_all
df_Nov_W_all[['WS-1','WS-2','WS-3','WS-25','WS-26','WS-27','WS-29']].iplot(size=(20,5),title='Wind Speed of 24h in Nov',xTitle='Time of Day', yTitle='Wind Speed,m/s')