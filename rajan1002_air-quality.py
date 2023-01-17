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

import seaborn as sns

from statsmodels.formula.api import ols
air=pd.read_csv("/kaggle/input/air-quality-data-brisbane-cbd/brisbanecbdaq2017.csv")
air.info()
air.replace("inf","NaN",inplace=True)

air.dropna(axis=0,how="any",inplace=True)
air["Date"]=pd.to_datetime(air.Date)

air=air.assign(day=air.Date.dt.day,month=air.Date.dt.month,year=air.Date.dt.year)

air.drop("Date",axis=1,inplace=True)
air.info()
air["day"].value_counts()
air["month"].value_counts()
air["year"].value_counts()
air.drop("year",axis=1,inplace=True)

air["Time"].value_counts()
air["Time"]=pd.to_datetime(air.Time)

air=air.assign(hour=air.Time.dt.hour)

air.drop("Time",axis=1,inplace=True)
corr_matrix=air.corr()

print(corr_matrix["PM10 (ug/m^3)"])
sns.heatmap(corr_matrix)
sns.lmplot(y="PM10 (ug/m^3)",x="Visibility-reducing Particles (Mm^-1)",data=air,fit_reg=False)
air.rename(columns={'Wind Direction (degTN)':'WD','Wind Speed (m/s)':'WS','Wind Sigma Theta (deg)':'WST','Wind Speed Std Dev (m/s)':'WSS','Air Temperature (degC)':'AT','Relative Humidity (%)':'RH','Rainfall (mm)':'RF','Barometric Pressure (hPa)':'BP','PM10 (ug/m^3)':'PM10','Visibility-reducing Particles (Mm^-1)':'VRP'},inplace=True)

model=ols('PM10~WD+WS+VRP+RH+AT+VRP*RH*WD-1',data=air).fit()

model.summary()