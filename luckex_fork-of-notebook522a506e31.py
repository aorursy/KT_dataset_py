# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Load the file

df = pd.read_csv('../input/HPI_master.csv')



# Describe dataset and basic statistcs

#print(df.head())

#print(df.tail())

print("===================================================================")



print("Dataframe size:\n", df.count())



print("===================================================================")





for x in df.columns:

    print(x,df[x].unique())



print("===================================================================")

print(df.describe())
import matplotlib.pyplot as plt



plt.style.use('ggplot')

#%matplotlib inline



fig, ax = plt.subplots(figsize=(6,4))

ax.hist(df['yr'], color='black')

ax.set_ylabel('Count per Year', fontsize=16)

ax.set_xlabel('Year', fontsize=16)

plt.title('Number of points per year', fontsize=18, y=1.01)
fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(df['yr'], df['index_nsa'], color='green')

ax.set_xlabel('Year', fontsize=16)

ax.set_ylabel('Price index', fontsize=16)

plt.title('Price index evolution over year', fontsize=18, y=1.01)
pd.pivot_table(df,index=["place_name"],values=["yr"],aggfunc=lambda x: len(x.unique())).sort_values("yr",ascending=False)
df.query('place_name == ["Florida"] and hpi_flavor == ["all-transactions"] and yr == 1975')
import datetime as dt

df_Florida = df.query('place_name == ["Florida"] and hpi_flavor == ["all-transactions"] and hpi_type == ["traditional"]')

df_Florida['date_intermediate'] = df_Florida['yr'].map(str) + "-" + df_Florida['period'].apply(lambda x: 1+3*(x - 1)).map(str)

df_Florida['date'] = df_Florida['date_intermediate'].apply(lambda x: pd.to_datetime(x, format='%Y-%m'))

del df_Florida['date_intermediate']

df_Florida.sort_values('date')

#df_Florida['internal_id'] = range(1, len(df_Florida) + 1)

df_Florida = df_Florida.reset_index()

df_Florida['internal_id'] = df_Florida.index

df_Florida.head()
plt.plot_date(df_Florida['date'], df_Florida['index_nsa'], xdate=True, ydate=False)
fig, ax1 = plt.subplots()

ax1.plot(df_Florida['index_nsa'], color='r')

ax1.set_xlabel('periods')

ax1.set_ylabel('index_nsa')

ax1.tick_params('y', colors='r')



import numpy

x = df_Florida['internal_id']

y = df_Florida['index_nsa']

dydx = numpy.diff(y) / numpy.diff(x)

df_Florida_Derivative = pd.DataFrame(dydx)



ax2 = ax1.twinx()

ax2.plot(df_Florida_Derivative, color='b')

ax2.set_ylabel('index_nsa derivative')

ax2.tick_params('y', colors='b')



fig.tight_layout()
import numpy

x = df_Florida['internal_id']

y = df_Florida['index_nsa']

dydx = numpy.diff(y) / numpy.diff(x)

df_Florida_Derivative = pd.DataFrame(dydx)



from scipy.signal import savgol_filter

dydx_smoothen = savgol_filter(dydx, 11, 1)

df_Florida_Derivative_smoothen = pd.DataFrame(dydx_smoothen)



df_Florida_Derivative_mean = [numpy.mean(dydx,0) for i in range(len(dydx))]



fig, ax1 = plt.subplots()

ax1.plot(df_Florida_Derivative, color='r')

ax1.plot(df_Florida_Derivative_smoothen, color='b')

ax1.plot(df_Florida_Derivative_mean, color='g')

ax1.set_xlabel('periods')

ax1.set_ylabel('index_nsa derivative')

ax1.tick_params('y', colors='r')

distance_mean = [dydx_smoothen[i] - numpy.mean(dydx,0) for i in range(len(dydx))]

threshold = 4*numpy.mean(dydx,0)

for i in range(len(distance_mean)):

    if distance_mean[i] < threshold and (i+1<len(distance_mean) and abs(distance_mean[i+1]) < threshold):

        distance_mean[i] = 0



flag=False

startIndex=0

stopIndex=0

PriceBubble=[]

PriceCrash=[]

for i in range(len(distance_mean)):

    if flag==False and distance_mean[i]!=0:

        flag=True

        startIndex=i

    if flag==True and distance_mean[i]==0:

        flag=False

        stopIndex=i-1

        if distance_mean[stopIndex] > 0:

            PriceBubble.append([startIndex, stopIndex])

        else:

            PriceCrash.append([startIndex, stopIndex])



print("Price Bubble:")

for i in range(len(PriceBubble)):

    print(str(df_Florida['date'][PriceBubble[i][0]])+"  /  "+str(df_Florida['date'][PriceBubble[i][1]]))

print("\n")

print("Price Crash:")

for i in range(len(PriceCrash)):

    print(str(df_Florida['date'][PriceCrash[i][0]])+"  /  "+str(df_Florida['date'][PriceCrash[i][1]]))