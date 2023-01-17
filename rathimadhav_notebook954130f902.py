# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "/kaggle/input/wind-turbine-scada-dataset/T1.csv"

df= pd.read_csv(path)
df.rename(columns={'Date/Time':'Time','LV ActivePower (kW)':'ActivePower(kW)',"Wind Speed (m/s)":"WindSpeed(m/s)","Wind Direction (°)":"Wind_Direction"},

                inplace=True)

sns.pairplot(df)
corr = df.corr()

plt.figure(figsize=(10, 8))



ax = sns.heatmap(corr, vmin = -1, vmax = 1, annot = True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()

corr
df.drop(['Wind_Direction'],axis=1,inplace = True)

df["Time"] = pd.to_datetime(df["Time"], format = "%d %m %Y %H:%M", errors = "coerce")

df
'''

import matplotlib.animation as animation

k=10000

curr=0

def update(curr):

    if curr == k: 

        a.event_source.stop()

    plt.cla()

    ax.plot(df['Time'][curr:100+curr],

            df['Theoretical_Power_Curve (KWh)'][curr:100+curr],

            color='purple',label = 'Predicted')



    ax.plot(df['Time'][curr:100+curr],

            df['ActivePower(kW)'][curr:100+curr],

            color='green',label = 'Actual')



    # Set title and labels for axes

    ax.set(ylabel="Theoretical_Power and Actual Power Curve",

           xlabel="Time",

           title="Time vs Power_Curve")



    plt.legend(loc = 'lower right',prop = {'size' : 15} )

    curr+=10

fig, ax = plt.subplots(figsize=(20,10))

a = animation.FuncAnimation(fig, update,interval =1000)

'''


y = df['Theoretical_Power_Curve (KWh)']

X = df[['ActivePower(kW)','WindSpeed(m/s)']]





'''

y = df['ActivePower(kW)']

X = df[['Theoretical_Power_Curve (KWh)','WindSpeed(m/s)']]

'''
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(max_leaf_nodes =500, random_state=1)

forest_model.fit(train_X, train_y)

power_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, power_preds))