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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from wordcloud import WordCloud

# display settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# initializing dataframe
df = pd.read_csv("/kaggle/input/phone-log/ImagePhoneLog.csv")
df["label"] = df["label"].replace(3,1)
df["label"] = df["label"].replace(2,0)
df["Ringer_RingerMode"] = df["Ringer_RingerMode"].replace(['Normal'],2)
df["Ringer_RingerMode"] = df["Ringer_RingerMode"].replace(['Silent'],0)
df["Ringer_RingerMode"] = df["Ringer_RingerMode"].replace(['Vibrate'],1)
df["Ringer_RingerMode"].value_counts()
df.shape
# column info
list(df.columns)
df.astype({'Ringer_RingerMode':'int64'})
df.info()
cor1 = df["label"].corr(df["Connectivity_isWifiAvailable"])
cor2 = df["label"].corr(df["Connectivity_isMobileAvailable"])
cor3 = df["label"].corr(df["Ringer_StreamVolumeVoicecall"])
cor4 = df["label"].corr(df["Battery_mBatteryLevel"])
cor5 = df["label"].corr(df["Battery_isCharging"])
cor6 = df["label"].corr(df["Ringer_RingerMode"])
cor7 = df["label"].corr(df["Battery_mBatteryPercentage"])
print(cor1,cor2,cor3,cor4,cor5,cor6,cor7)
#print(cor2)
df.isna().sum()
#df["AppUsage_LastestForegroundPackage"].dropna().head()
# no. of rows in each rating bracket
df["Accessibility_pack"].dropna().value_counts()
df["Accessibility_pack"].describe()
#df.isna().sum()
