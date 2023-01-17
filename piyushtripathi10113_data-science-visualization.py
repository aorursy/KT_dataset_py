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
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
df_pgen1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
df_sen1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
df_pgen2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
df_sen2 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 100
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_sen1['DATE_TIME'] = pd.to_datetime(df_sen1['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_sen1['DATE'] = df_sen1['DATE_TIME'].apply(lambda x:x.date())
df_sen1['TIME'] = df_sen1['DATE_TIME'].apply(lambda x:x.time())
df_sen1['DATE'] = pd.to_datetime(df_sen1['DATE'],format = '%Y-%m-%d')
df_sen1['HOUR'] = pd.to_datetime(df_sen1['TIME'],format='%H:%M:%S').dt.hour
df_sen1['MINUTES'] = pd.to_datetime(df_sen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute
df_sen2['DATE_TIME'] = pd.to_datetime(df_sen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_sen2['DATE'] = df_sen2['DATE_TIME'].apply(lambda x:x.date())
df_sen2['TIME'] = df_sen2['DATE_TIME'].apply(lambda x:x.time())
df_sen2['DATE'] = pd.to_datetime(df_sen2['DATE'],format = '%Y-%m-%d')
df_sen2['HOUR'] = pd.to_datetime(df_sen2['TIME'],format='%H:%M:%S').dt.hour
df_sen2['MINUTES'] = pd.to_datetime(df_sen2['TIME'],format='%H:%M:%S').dt.minute
iplot([go.Histogram2dContour(x=df_pgen1.head(10000)['HOUR'], 
                             y=df_pgen1.head(10000)['DC_POWER'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen1.head(20000)['HOUR'], y=df_pgen1.head(20000)['DC_POWER'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen1.head(10000)['HOUR'], 
                             y=df_pgen1.head(10000)['AC_POWER'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen1.head(20000)['HOUR'], y=df_pgen1.head(20000)['AC_POWER'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen1.head(10000)['HOUR'], 
                             y=df_pgen1.head(10000)['DAILY_YIELD'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen1.head(20000)['HOUR'], y=df_pgen1.head(20000)['DAILY_YIELD'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen1.head(10000)['HOUR'], 
                             y=df_pgen1.head(10000)['TOTAL_YIELD'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen1.head(20000)['HOUR'], y=df_pgen1.head(20000)['TOTAL_YIELD'], mode='markers')])
iplot([go.Histogram2dContour(x=df_sen1.head(10000)['HOUR'], 
                             y=df_sen1.head(10000)['AMBIENT_TEMPERATURE'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_sen1.head(20000)['HOUR'], y=df_sen1.head(20000)['AMBIENT_TEMPERATURE'], mode='markers')])
iplot([go.Histogram2dContour(x=df_sen1.head(10000)['HOUR'], 
                             y=df_sen1.head(10000)['MODULE_TEMPERATURE'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_sen1.head(20000)['HOUR'], y=df_sen1.head(20000)['MODULE_TEMPERATURE'], mode='markers')])
iplot([go.Histogram2dContour(x=df_sen1.head(10000)['HOUR'], 
                             y=df_sen1.head(10000)['IRRADIATION'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_sen1.head(20000)['HOUR'], y=df_sen1.head(20000)['IRRADIATION'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen2.head(10000)['HOUR'], 
                             y=df_pgen2.head(10000)['DC_POWER'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen2.head(20000)['HOUR'], y=df_pgen2.head(20000)['DC_POWER'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen2.head(10000)['HOUR'], 
                             y=df_pgen2.head(10000)['AC_POWER'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen2.head(20000)['HOUR'], y=df_pgen2.head(20000)['AC_POWER'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen2.head(10000)['HOUR'], 
                             y=df_pgen2.head(10000)['DAILY_YIELD'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen2.head(20000)['HOUR'], y=df_pgen2.head(20000)['DAILY_YIELD'], mode='markers')])
iplot([go.Histogram2dContour(x=df_pgen2.head(10000)['HOUR'], 
                             y=df_pgen2.head(10000)['TOTAL_YIELD'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_pgen2.head(20000)['HOUR'], y=df_pgen2.head(20000)['TOTAL_YIELD'], mode='markers')])
iplot([go.Histogram2dContour(x=df_sen2.head(10000)['HOUR'], 
                             y=df_sen2.head(10000)['AMBIENT_TEMPERATURE'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_sen2.head(20000)['HOUR'], y=df_sen2.head(20000)['AMBIENT_TEMPERATURE'], mode='markers')])
iplot([go.Histogram2dContour(x=df_sen2.head(10000)['HOUR'], 
                             y=df_sen2.head(10000)['MODULE_TEMPERATURE'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_sen2.head(20000)['HOUR'], y=df_sen2.head(20000)['MODULE_TEMPERATURE'], mode='markers')])
iplot([go.Histogram2dContour(x=df_sen2.head(10000)['HOUR'], 
                             y=df_sen2.head(10000)['IRRADIATION'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=df_sen2.head(20000)['HOUR'], y=df_sen2.head(20000)['IRRADIATION'], mode='markers')])
sns.pairplot(df_pgen1.iloc[:,3:7]);
sns.pairplot(df_pgen1.iloc[:,3:5],diag_kind="kde", markers="", kind='reg');
plt.show()
sns.pairplot(df_sen1.iloc[:,3:6]);
sns.pairplot(df_sen1.iloc[:,3:6],diag_kind="kde", markers="", kind='reg');
plt.show()
sns.pairplot(df_pgen2.iloc[:,3:7]);
sns.pairplot(df_pgen2.iloc[:,3:5],diag_kind="kde", markers="", kind='reg');
plt.show()
sns.pairplot(df_sen2.iloc[:,3:6]);
sns.pairplot(df_sen2.iloc[:,3:6],diag_kind="kde", markers="", kind='reg');
plt.show()