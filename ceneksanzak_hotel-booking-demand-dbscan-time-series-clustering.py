import numpy as np 

import pandas as pd 

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

import seaborn as sn

import matplotlib.pyplot as plt
df.head()
df["arrival_date"] = df.apply(lambda d: pd.Timestamp(str(d["arrival_date_day_of_month"]) + " " + d["arrival_date_month"] + " " + str(d["arrival_date_year"])), axis=1)
df2 = pd.DataFrame( index=df.arrival_date.unique()).sort_index()

df2["daily_booking"] = df["arrival_date"].value_counts()

df2["days_in_waiting_list"] = df[["arrival_date", "days_in_waiting_list"]].groupby("arrival_date").mean()

df2["total_of_special_requests"] = df[["arrival_date", "total_of_special_requests"]].groupby("arrival_date").mean()

df2["adr"] = df[["arrival_date", "adr"]].groupby("arrival_date").mean()

df2["booking_changes"] = df[["arrival_date", "booking_changes"]].groupby("arrival_date").mean()

df2["previous_cancellations"] = df[["arrival_date", "previous_cancellations"]].groupby("arrival_date").mean()

df2["stays_in_weekend_nights"] = df[["arrival_date", "stays_in_weekend_nights"]].groupby("arrival_date").mean()

df2["stays_in_week_nights"] = df[["arrival_date", "stays_in_week_nights"]].groupby("arrival_date").mean()

columns = ["daily_booking", "days_in_waiting_list", "total_of_special_requests", "adr", "booking_changes", "previous_cancellations"]

df2[columns] = StandardScaler().fit_transform(df2[columns])
df2.describe()
from sklearn.cluster import DBSCAN



clustering = DBSCAN(eps=1.10, min_samples=3).fit(df2[columns])

df2["cluster"] = clustering.labels_

df2["cluster"].value_counts()
train = pd.DataFrame(df2.iloc[:-100])

test = pd.DataFrame(df2.iloc[-100:])
for i in columns:

    fig, ax = plt.subplots(figsize=(20,6))



    a = test.loc[test['cluster'] == -1].index

    b= test[(test['cluster'] == -1)][i]

    ax.scatter(a, b, color='red', label='Anomaly',s = 200)

    ax.plot(train.index, train[i], color='blue', label='Train',linewidth=0.7)

    ax.plot(test.index, test[i], color='green', label='Test',linewidth=0.7)

    plt.xlabel('Arrival Date')

    plt.ylabel(i)

    plt.legend()

    plt.show();
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(df2[columns])

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])
import matplotlib.dates as md

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



colors = {0:'black', 1:'blue', 2:'green', -1: 'red'}

for i in range(-1,3):

    ax.scatter(principalDf['principal component 1'].iloc[-100:], principalDf['principal component 2'].iloc[-100:], c=test["cluster"].apply(lambda x: colors[x if x<3 else 0]))

plt.show();
corrMatrix = df2.corr()

plt.figure(figsize = (10,10))

sn.heatmap(corrMatrix, annot=True)

plt.show()

corrMatrix = df2.corr(method="kendall")

plt.figure(figsize = (10,10))

sn.heatmap(corrMatrix, annot=True)

plt.show()
