import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/548909889_2020_03_28.csv", parse_dates=True,

                infer_datetime_format = True)

df["session_begin"] = df["session_begin"].apply(lambda x: pd.to_datetime(x))

df["session_end"] = df["session_end"].apply(lambda x: pd.to_datetime(x))

#df['duration_manual']=(df["session_end"]-df["session_begin"]).apply(lambda x: x.seconds//60)
df["session_begin_during_day_hours"]=df["session_begin"].apply(lambda x: x.time().hour)

df["session_end_during_day_hours"]=df["session_end"].apply(lambda x: x.time().hour)

df["session_begin_during_day_minutes"]=df["session_begin"].apply(lambda x: x.time().hour*60+x.time().minute)

df["session_end_during_day_minutes"]=df["session_end"].apply(lambda x: x.time().hour*60+x.time().minute)

df.hist()

plt.show()

sessions_held_on_hour=[]

for i in range(25):

    sessions_held_on_hour.append(len(df[(df["session_begin_during_day_hours"]< i) &

    (df["session_end_during_day_hours"]> i)])) 

sessions_held_on_minutes=[]

for i in range(24*60+60):

    sessions_held_on_minutes.append(len(df[(df["session_begin_during_day_minutes"]< i) &

    (df["session_end_during_day_minutes"]> i)])) 

from matplotlib import pyplot as plt

plt.plot(range(25), sessions_held_on_hour)

plt.show()

plt.plot(range(24*60+60), sessions_held_on_minutes)

plt.show()
import math

df.corr()[["session_duration"]].apply(lambda x: abs(x/(1-x*x)**0.5)*(len(df)-2))
df["session_duration"].hist()

print(df.columns)

df1 = df[['session_duration',

       'session_begin_during_day_hours', 'session_end_during_day_hours']]

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X1 = df1[['session_begin_during_day_hours']]

X2 = df1[['session_end_during_day_hours']]

y =  df1['session_duration']

X1_train, X1_test, y_train, y_test = train_test_split(X1, y)

model = LinearRegression()

model.fit(X1_train, y_train)

print(model.score(X1_test,y_test))

print(model.coef_)

X2_train, X2_test, y_train, y_test = train_test_split(X2, y)

model = LinearRegression()

model.fit(X2_train, y_train)

print(model.score(X2_test,y_test))

print(model.coef_)