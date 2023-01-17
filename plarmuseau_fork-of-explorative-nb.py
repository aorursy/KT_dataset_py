import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

outlet = pd.read_csv("../input/outlets.csv")

outlet.describe().T
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

import matplotlib

import matplotlib.pyplot as plt



loc_df = pd.DataFrame()

loc_df['longitude'] = outlet.Longitude

loc_df['latitude'] = outlet.Latitude

kmeans = KMeans(n_clusters=165, random_state=1, n_init = 10).fit(loc_df)

Day_distancematrix=pd.DataFrame(kmeans.transform(loc_df))

clust_cent=kmeans.cluster_centers_



loc_df['label'] = kmeans.labels_

outlet['cluster_day']=kmeans.labels_

plt.figure(figsize = (18,12))

for label in loc_df.label.unique():

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize =2)

    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')

    plt.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')

    plt.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 10)





plt.title('Day Cluster Belgium')

plt.show()





kmeans = KMeans(n_clusters=82, random_state=1, n_init = 10).fit(loc_df)

Night_distancematrix=pd.DataFrame(kmeans.transform(loc_df))

loc_df['label'] = kmeans.labels_

outlet['cluster_night']=kmeans.labels_



# temp

min_day_distance=Day_distancematrix.min()

for xm in range(len(min_day_distance)):

    temp=Day_distancematrix[Day_distancematrix[xm].isin(min_day_distance)].index[0]

    



    
# create empty guard grid

import datetime  

tindex=pd.date_range(start='2018-01-01 9:00', end='2018-12-31 9:00')

calender=pd.DataFrame(np.zeros((365, 165)),tindex)

calender.reset_index(inplace=True)



#fill in with randomseed to create a starting situation with the saldo from previous year

from random import randint

print(randint(0, 9))

outlet['Capital']=[randint(0,1) for x in range(len(outlet))]

#print(calender)

#select all available pharmacy's

for xd in range(len(calender)):

    #drop 50% pharmacy with highest capital

    gemiddelde=np.mean(outlet['Capital'])

    #print(gemiddelde)

    outlet_sql1=outlet[outlet['Capital']<gemiddelde]

    # drop pharmacy with blackpoints or vacations

    dag=calender.iloc[xd]['index']

    jaar,week,dag=dag.isocalendar()

    outlet_sql2=outlet_sql1[outlet_sql1['vacationweek1'] != week ]

    # drop pharmacy clusters already selected due to week scheme

    # sort lowest guard

    outlet_sql3=outlet_sql2.sort_values(['cluster_day','Capital'])

    #print(outlet_sql3)

    kmeans = KMeans(n_clusters=165,init=clust_cent, random_state=1, n_init = 10,n_jobs=-2).fit(outlet_sql3[['Longitude','Latitude']])

    if round(xd/10)==xd:

        print(xd)    

    Day_distancematrix=pd.DataFrame(kmeans.transform(outlet_sql3[['Longitude','Latitude']]))

    clust_cent=kmeans.cluster_centers_

    min_day_distance=Day_distancematrix.min()

    for xm in range(len(min_day_distance)):

        temp=Day_distancematrix[Day_distancematrix[xm].isin(min_day_distance)]



        temp2=temp.reset_index()

        temp3=np.int(temp2.iloc[0]['index'])

        pharmid=outlet_sql3.iat[temp3,0]

        calender.iat[xd,xm+1]=pharmid

        outlet.iat[pharmid,3]=outlet.iat[pharmid,3]+2.0

        #print(temp3,outlet.iloc[temp3]['Capital'],outlet.iat[temp3,3])



print(calender)

# plot distribution of Capital

import matplotlib.pyplot as plt

plt.hist(outlet['Capital'],26)

plt.title("Histogram of Capital points of all pharmacies")

plt.xlabel("Value")

plt.ylabel("Frequency")

plt.show()