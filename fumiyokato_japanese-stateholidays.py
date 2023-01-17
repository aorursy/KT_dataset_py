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
cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

stv = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
# sales_train_validationの行を日系列にする

TMP_stv = stv.set_index("id")

TMP_stv = TMP_stv.iloc[:, 5:].T

TMP_stv.reset_index(inplace=True)

TMP_stv.rename(columns={"index":"d"}, inplace=True)

print(TMP_stv.shape)

TMP_stv.head()
# カレンダー情報calと販売数情報TMP_stvをマージしてひとつのDFにする

cal_stv = pd.merge(cal, TMP_stv, on="d")

print(cal_stv.shape)

cal_stv.head()
# CaliforniaのみのDF "CAdf"

CAcol = cal_stv.loc[:, cal_stv.columns.str.contains("CA")]



df = cal_stv.loc[:, 

                 ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d", "event_name_1", "event_type_1"]]



CAdf = pd.concat([df, CAcol], axis=1)

CAdf.head()
from datetime import datetime as dt



# "date"列を年月日で読み込む

CAdf["date"] = pd.to_datetime(CAdf["date"])



# State holidayの日の"wday"を0にする

CAdf.loc[CAdf["date"].apply(lambda x: x in [dt(2011,3,31), dt(2012,3,31), dt(2013,3,31), dt(2014,3,31), 

                                        dt(2015,3,31), dt(2016,3,31), dt(2015,11,27), dt(2011,11,25)]), 

       "wday"] = 0



CAdf[CAdf["date"] == dt(2011,3,31)]
CAdf.loc[(~CAdf["event_name_1"].isnull()), "wday"] = 0

CAdf[CAdf["wday"] == 0]