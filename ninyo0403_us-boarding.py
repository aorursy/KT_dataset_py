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
df = pd.read_csv(os.path.join(dirname, filename))
df
list(set(df['Measure']))
df
df_group = df[['Port Name','Value','Measure']].groupby(by=['Port Name','Measure']).sum()
df_group.sort_values(by='Value').tail(10).plot.barh()
df_group.sort_values(by=['Value'],ascending=False).head(10)
df[['Port Name','Value','Measure']].groupby(by='Port Name').sum().sort_values(by='Value',ascending=False).head(10)
df[['Value','Measure']].groupby(by=['Measure']).sum().sort_values(by='Value',ascending=False)
df[df['Measure'] == 'Personal Vehicle Passengers'][['Port Name','Value','Date']].groupby(by=['Port Name']).sum().sort_values(by='Value',ascending=True).tail(5).plot.barh()
df[df['Measure'] == 'Personal Vehicle Passengers'][['Port Name','Value','Date']].groupby(by=['Port Name']).sum().sort_values(by='Value',ascending=True).tail(5)
df[df['Port Name'] == 'El Paso'][['Value','Measure']].groupby(by='Measure').sum().sort_values(by='Value',ascending=True).plot.barh()
df[df['Port Name'] == 'El Paso'][df['Measure'] == 'Personal Vehicle Passengers']
df[df['Port Name'] == 'El Paso'][df['Measure'] == 'Personal Vehicle Passengers'].sort_index(ascending=False).plot(x='Date',y='Value')
from datetime import datetime
datetime.strptime(df['Date'][0], '%m/%d/%Y %H:%M')
df['Date2'] = 0
for i in range(len(df)):

    df['Date2'][i] = datetime.strptime(df['Date'][i], '%m/%d/%Y %H:%M')
df[df['Date2'] > datetime(1999,12,1,0,0)][['Port Name','Value','Measure']].groupby(by='Port Name').sum().sort_values(by='Value',ascending=False).head(10)
df[df['Date2'] > datetime(2009,12,1,0,0)][df['Measure'] == 'Personal Vehicle Passengers'][['Port Name','Value','Measure']].groupby(by='Port Name').sum().sort_values(by='Value',ascending=False).head(10)
df[df['Port Name'] == 'San Ysidro'][df['Measure'] == 'Personal Vehicle Passengers'].sort_index(ascending=False).plot(x='Date',y='Value')