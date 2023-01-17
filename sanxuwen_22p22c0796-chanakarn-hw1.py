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
df  = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df['DATE_TIME'] = pd.to_datetime(df["DATE_TIME"])
df.head(5)
df.groupby
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df2['DATE_TIME'] = pd.to_datetime(df2["DATE_TIME"])

df2 = df2[['DATE_TIME','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']]
df2.head(5)
df = pd.merge(df,df2,how = 'inner',on="DATE_TIME")
df['DATE'] = df['DATE_TIME'].dt.date
df
temp = df.groupby(['DATE']).mean().reset_index()[['DATE','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']]
temp2 = df.groupby(['DATE']).sum().reset_index()[['DATE','DC_POWER','AC_POWER','DAILY_YIELD']]
df_final = pd.merge(temp,temp2,how = 'inner',on = 'DATE')
df_final.shape
df_final
df_final[['AMBIENT_TEMPERATURE_SHIFT3','MODULE_TEMPERATURE_SHIFT3','IRRADIATION_SHIFT3','DC_POWER_SHIFT3','AC_POWER_SHIFT3']] = df_final[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER','AC_POWER']].shift(periods=3)
df_final_shift3 = df_final[['AMBIENT_TEMPERATURE_SHIFT3','MODULE_TEMPERATURE_SHIFT3','IRRADIATION_SHIFT3','DC_POWER_SHIFT3','AC_POWER_SHIFT3','DAILY_YIELD']]
df_final_shift3 = df_final_shift3.dropna()
df_final_shift3 = df_final_shift3.reset_index(drop = True)
df_final_shift3
splits = []

for i in range(10):

    splits.append(df_final_shift3.iloc[i::10,])
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



for fold in range(10):

    df_val = splits[fold]

    df_train = pd.concat((splits[:fold] + splits[fold+1:])).sort_index()

    X_train = df_train[['AMBIENT_TEMPERATURE_SHIFT3','MODULE_TEMPERATURE_SHIFT3','IRRADIATION_SHIFT3','DC_POWER_SHIFT3','AC_POWER_SHIFT3']]

    y_train = df_train[['DAILY_YIELD']]

    X_val = df_val[['AMBIENT_TEMPERATURE_SHIFT3','MODULE_TEMPERATURE_SHIFT3','IRRADIATION_SHIFT3','DC_POWER_SHIFT3','AC_POWER_SHIFT3']]

    y_val = df_val[['DAILY_YIELD']]

    lr = LinearRegression()

    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_val)

#     print(y_val)

    print(np.sqrt(mean_squared_error(y_pred,y_val)))

    



    
df_final[['AMBIENT_TEMPERATURE_SHIFT7','MODULE_TEMPERATURE_SHIFT7','IRRADIATION_SHIFT7','DC_POWER_SHIFT7','AC_POWER_SHIFT7']] = df_final[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER','AC_POWER']].shift(periods=7)
df_final_shift7 = df_final[['AMBIENT_TEMPERATURE_SHIFT7','MODULE_TEMPERATURE_SHIFT7','IRRADIATION_SHIFT7','DC_POWER_SHIFT7','AC_POWER_SHIFT7','DAILY_YIELD']]
df_final_shift7 = df_final_shift7.dropna()
df_final_shift7 = df_final_shift7.reset_index(drop = True)
splits = []

for i in range(10):

    splits.append(df_final_shift7.iloc[i::10,])
for fold in range(10):

    df_val = splits[fold]

    df_train = pd.concat((splits[:fold] + splits[fold+1:])).sort_index()

    X_train = df_train[['AMBIENT_TEMPERATURE_SHIFT7','MODULE_TEMPERATURE_SHIFT7','IRRADIATION_SHIFT7','DC_POWER_SHIFT7','AC_POWER_SHIFT7']]

    y_train = df_train[['DAILY_YIELD']]

    X_val = df_val[['AMBIENT_TEMPERATURE_SHIFT7','MODULE_TEMPERATURE_SHIFT7','IRRADIATION_SHIFT7','DC_POWER_SHIFT7','AC_POWER_SHIFT7']]

    y_val = df_val[['DAILY_YIELD']]

    lr = LinearRegression()

    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_val)

#     print(y_val)

    print(np.sqrt(mean_squared_error(y_pred,y_val)))

    



    
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=10)

# for train_index, test_index in kf.split(df):

#     print("TRAIN:", train_index, "TEST:", test_index)

#     X_train, X_test = df.iloc[train_index], df.iloc[test_index]

#     y_train, y_test = df.iloc[train_index], df.iloc[test_index]

# X_train