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
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
file_name = '/kaggle/input/hdemrep35_113.xlsx'
sheet_name = 'Sheet1'
df_raw = pd.read_excel(file_name, sheet_name = sheet_name, header = None, charset = 'utf-8')
column_names = ['Congress_number', 'Roll_Call_number', 'Month', 'Day', 'Year', 'Number_Missing_Votes', 
                'Number_Yeas', 'Number_Nays', 'Number_Republican_Yeas', 'Number_Republican_Nays',
               'Number_Democrat_Yeas', 'Number_Democrat_Nays', 'Number_Northern_Republican_Yeas',
               'Number_Northern_Republican_Nays', 'Number_Southern_Republican_Yeas', 'Number_Southern_Republican_Nays',
               'Number_Northern_Democrat_Yeas', 'Number_Northern_Democrat_Nays', 'Number_Southern_Democrat_Yeas',
               'Number_Southern_Democrat_Nays']
df_raw.columns = column_names
import datetime
# df_raw['Date'] = datetime.datetime(df_raw['Year'], df_raw['Month'], df_raw['Day'])
df_raw['Date']  = df_raw.apply(lambda row: datetime.datetime(row.Year, row.Month, row.Day), axis = 1)
df_raw['Weekday']  = df_raw.apply(lambda row: row.Date.weekday() , axis = 1)
df_raw['Total_members']  = df_raw.apply(lambda row: row.Number_Missing_Votes + row.Number_Yeas + row.Number_Nays , axis = 1)
df_raw.head()
for col in column_names[5:]:
    df_raw[col + '_Norm']  = np.round(df_raw[col]/df_raw["Total_members"], 3)
df_raw.tail()  
df_raw.columns #23
col_names_updated = list(df_raw.columns)
col_names_updated[23:]
df_raw.loc[:, 'Number_Missing_Votes_Norm': 'Number_Southern_Democrat_Nays_Norm'].hist(bins=50, figsize=(20,15), color = 'green') 
def boxPlot(x, y, data):
    fig, ax = plt.subplots(figsize=(50,15))
    ax1 = sns.boxplot(ax=ax, x=x, y=y, data=data)
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
    plt.plot()
    return None
for y in col_names_updated[23:]:
    boxPlot(x='Year', y=y, data=df_raw)
for y in col_names_updated[23:]:
    boxPlot(x='Congress_number', y=y, data=df_raw)
for y in col_names_updated[23:]:
    boxPlot(x='Day', y=y, data=df_raw)
for y in col_names_updated[23:]:
    boxPlot(x='Weekday', y=y, data=df_raw) #Return the day of the week as an integer, where Monday is 0 and Sunday is 6.