# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv('/kaggle/input/corona-virus-dataset/covid_19_data.csv')
df.drop('SNo', axis=1, inplace = True)
# df.set_index('ObservationDate', inplace=True)
df.head(10)
df.isnull().sum()
df.fillna(method='ffill', inplace=True)
df.isnull().sum()
df.shape
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df.set_index('ObservationDate', inplace=True)
df
import matplotlib.pyplot as plt

import seaborn as sns
query_01 = df.Confirmed.resample('D').sum()

print(query_01)
plt.figure(figsize=(15,10))

plt.title('Changes in Number of affected cases over time')

plt.xlabel('TIME')

plt.ylabel('Number of Cases')

query_01.plot(kind='bar', stacked = True)
# plt.figure(figsize=(40,40))

query_02 = df.groupby(['Country/Region']).sum()
query_02.reset_index(inplace=True)
query_02
# Plot

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

# plt.figure(figsize=(20,10) )

# plt.title('Changes in cases over time at country level')



wedges, texts, c= ax.pie(query_02['Confirmed'], autopct='%1.1f%%', wedgeprops=dict(width=0.5), startangle=-40)



bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax.annotate(query_02['Country/Region'][i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



ax.set_title("Changes in cases over time at country level")



plt.show()
df.columns
df['Last Update'] = pd.to_datetime(df['Last Update'])
df.Confirmed.resample('D').sum().to_frame().tail(20).plot(kind='bar')
query_04 = df.groupby('Country/Region').sum()
query_04
query_04.reset_index(inplace=True)
# Plot

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

# plt.figure(figsize=(20,10) )

# plt.title('Changes in cases over time at country level')



wedges, texts, c= ax.pie(query_04['Deaths'], autopct='%1.1f%%', wedgeprops=dict(width=0.5), startangle=-40)



bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax.annotate(query_04['Country/Region'][i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



ax.set_title("Country wise reported death cases till latest date")



plt.show()
query_05 = df.groupby('Country/Region').sum()
query_05
query_05.reset_index(inplace=True)
# Plot

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))

# plt.figure(figsize=(20,10) )

# plt.title('Changes in cases over time at country level')



wedges, texts, c= ax.pie(query_05['Recovered'], autopct='%1.1f%%', wedgeprops=dict(width=0.5), startangle=-40)



bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax.annotate(query_05['Country/Region'][i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                horizontalalignment=horizontalalignment, **kw)



ax.set_title("Country wise recovered cases till latest date")



plt.show()