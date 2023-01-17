# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
!ls -GFlash --color ../input/m5-forecasting-accuracy/
# Read in the data
INPUT_DIR = '../input/m5-forecasting-accuracy'
cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')
ste = pd.read_csv(f'{INPUT_DIR}/sales_train_evaluation.csv')
ss.head()
ste.head()
stv.head()
d_cols = [c for c in ste.columns if 'd_' in c]
ste_tmp=ste.set_index('id')[d_cols]\
    .T
ste_tmp['d']=ste_tmp.index
ste_tmp.head()
date=ste_tmp.index.values
print(date)
ste_cal=ste_tmp.merge(cal, left_on='d', right_on='d',how='outer')
df=ste_cal
df_x=df[['wm_yr_wk','wday']]
print(df_x.query('index == "1940"'))

#from sklearn.model_selection import train_test_split
#x_train, x_test = train_test_split(df_x, train_size=0.6, shuffle=False)
x_train=df_x[:1941]
x_test=df_x[1941:]
print(x_train)
print(x_test)
#yosousurudake
from sklearn.linear_model import LinearRegression as LR 
predict_y=[]
for i in range(0,30489):
    df_y=df.iloc[:,[i]]
    y_train=df_y[:1941]
    model1 = LR()
    model1.fit(x_train, y_train)
    predict_y.append(model1.predict(x_test).tolist())
print(predict_y[0])
print(len(predict_y))
print(ss.loc[0:0, 'F1': 'F28'])
print(predict_y[0])
for i in range(0,len(predict_y)):
    predict_y[i] = [float(x) for y in predict_y[i] for x in y]
    

#predict_y[0] = [float(x) for y in predict_y[0] for x in y]  
print(predict_y[2])
ss.head()
f_cols = [c for c in ss.columns if 'F' in c]
ss_tmp=ss.set_index('id')[f_cols]\
    .T
ss_tmp['f']=ss_tmp.index
ss_tmp.columns
ss_tmp.head()
ste_columns=stv['id']
#ste_columns=ste_tmp.columns
forecast_ste=pd.DataFrame(columns=ste_columns)
print(ste_columns)
for i in range(0, len(predict_y)):
    forecast_ste.loc[:, forecast_ste.columns[i]]=predict_y[i]
    
forecast_ste.head()
forecast=forecast_ste.T
forecast.reset_index()
forecast['id'] = forecast.index
forecast.columns
forecast=forecast.rename(columns={'id': 'ID'})
forecast['ID']
#ste_cal=ste_tmp.merge(cal, left_on='d', right_on='d',how='outer')
new_ss=ss.merge(forecast, left_on='id', right_on='ID', how='left')
new_ss.head()
for i in range(0,28):
    new_ss['F'+str(i+1)]=new_ss[i]
    del new_ss[i]
del new_ss['ID']
new_ss.head()
new_ss=new_ss.fillna(0)
new_ss.to_csv('submission.csv', index=False)