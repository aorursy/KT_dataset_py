# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

%matplotlib inline

from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/crime.csv", 

                   nrows=7000

                  ,parse_dates=['LAST_OCCURRENCE_DATE', 'FIRST_OCCURRENCE_DATE'])

df=pd.DataFrame(data)
police_colors = ['#356CB5',  # Steel Blue

                 '#1B51A1',  # Cyan Cobalt Blue

                 '#1B51A1',

                 '#0F3C8B',  # Dark Cerulean

                 '#590000',

                 '#980001',  # Crimson Red

                 '#CC0000',  # Boston University Red 

                 '#E82734',  # Alizarin Crimson

                 ]

my_cmap = ListedColormap(sns.color_palette(police_colors).as_hex())



sns.set_palette(palette=police_colors)

sns.set_style("darkgrid")

sns.palplot(police_colors)
#plotting nulls

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=my_cmap)
#Grouping offense types

CountofReports=df.groupby(df['OFFENSE_CATEGORY_ID']).size()

CountofLOD=df.groupby(df['OFFENSE_CATEGORY_ID'])['LAST_OCCURRENCE_DATE'].count()



#Finding a difference

Difference=CountofReports-CountofLOD



#plotting

fig,(ax1,ax2,ax3)= plt.subplots(figsize=(22,6),nrows=1,ncols=3)

CountofReports.plot(kind='bar',ax=ax1,cmap=my_cmap, title="Count of Reports")

CountofLOD.plot(kind ='bar',ax=ax2,cmap=my_cmap,title="Count of LAST_OCCURRENCE_DATE")

Difference.plot(kind='bar',ax=ax3,cmap=my_cmap,title='Difference')
#Rename DateTime fields for better understanding of what's going on.

df.rename(columns={'FIRST_OCCURRENCE_DATE': 'BeginDate', 'LAST_OCCURRENCE_DATE': 'EndDate'}, inplace=True)



#replacing null LOD dates with FOD when LOD is null

df.EndDate.fillna(df.BeginDate,inplace=True)



#removing Lat,Long combos with missing data

df.dropna(subset=['GEO_X'],inplace=True)
#finding hour1

df['Hour1']= df.BeginDate.dt.hour



#finding hour2 given comparative circumstances

df['Hour2']=np.where(

    (df['BeginDate'].dt.hour == df['EndDate'].dt.hour),

        df['EndDate'].dt.hour,

            np.where((df['EndDate'].dt.hour == 0) & (df['EndDate'].dt.minute<10),

                     23,

                         np.where((df['EndDate'].dt.minute<10),

                                  (df['EndDate'].dt.hour - 1),

                                      df['EndDate'].dt.hour)))



#creating an hours in range based on the two fields seen above

df['HoursInRange']= np.where((df['Hour1']>df['Hour2']), 25-(df['Hour1']-df['Hour2']),(df['Hour2']-df['Hour1']+1))



#filtering out all crimes with an Hours in Range >24 and renaing the dataframe

crimes=df[df['HoursInRange']<=24]



#creating the midpoint

crimes['SplitDate'] = crimes['BeginDate'] + (crimes["EndDate"] - crimes['BeginDate'])/2

crimes['SplitDate'] = crimes['SplitDate'].apply(lambda x: x.strftime('%m/%d/%Y %H:00:00'))

crimes['SplitDate'] = pd.to_datetime(crimes.SplitDate)
def Heatmap_daysXhours(data_frame, groupby_field, count_field, off):

    heat_group=data_frame.groupby(data_frame[groupby_field], as_index=True, sort=True)[count_field].count()

    heat_data=pd.DataFrame(heat_group)

    heat_data['Hours']= heat_data.index.hour

    heat_data['Days']= heat_data.index.day_name()

    heat_data_pivot=pd.pivot_table(heat_data,values=count_field, index='Days',columns='Hours',fill_value=0)

    

    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

    fig,(ax,cbar_ax)= plt.subplots(2,figsize=(15,7),gridspec_kw=grid_kws)

    ax.set_title(off+' by Time of Day and Day of Week',fontsize=18)

    cbar_ax.set_title('Count of Police Reports',fontsize=12)

    ax = sns.heatmap(heat_data_pivot, square=True,

                     cmap=my_cmap, annot=True,

                     linewidths=.01, ax=ax,

                     cbar_ax=cbar_ax, 

                     cbar_kws={"orientation": "horizontal"})

    ax.set_ylabel("Days of the Week", fontsize=15)

    ax.set_xlabel("24 Hours of the Day", fontsize=15)
vehicleburglary=crimes[crimes['OFFENSE_CATEGORY_ID']=='theft-from-motor-vehicle']

Heatmap_daysXhours(vehicleburglary,'SplitDate','INCIDENT_ID','Vehicle Burglaries')
accidents=crimes[crimes['OFFENSE_CATEGORY_ID']=='traffic-accident']

Heatmap_daysXhours(accidents,'SplitDate','INCIDENT_ID','Accidents')
#creating a weight function and applying the function to a new column

def Weight(Span): 

    try: 

        return round(1/Span,3) 

    except ZeroDivisionError: return 1



#applying the weights to a new column    

crimes['Weight']=crimes['HoursInRange'].apply(Weight)





#Creating 24 columns, which are hours of the day, and appling the algo's results for each case to each applicable hour



#Previously I was doing this for 24 hours in a day

#df['01:00'] = np.where((df['Hour2'] >= df['Hour1']),

                    #np.where((1>=df['Hour1']) & (1<=df['Hour2']),df['Weight'],0),

                        #np.where( (1<=df['Hour2']) | (1>=df['Hour1']),df['Weight'],0))



#now I'm doing this instead of what's above here.

x=np.arange(0,24,1)

for i in x:

    crimes[i]=np.where((crimes['Hour2'] >= crimes['Hour1']),

                       np.where((i>=crimes['Hour1']) & (i<=crimes['Hour2']),crimes['Weight'],0),

                           np.where( (i<=crimes['Hour2']) | (i>=crimes['Hour1']),crimes['Weight'],0))



crimes.head()
def Aoristic_Chart(data,chart_offense):

    ax = data.iloc[20:44].plot(

        xticks=np.arange(0,24,1),

        grid=True,

        figsize=(15,7),

        fontsize=16)

    ax.set_xlim([0,23])

    ax.set_ylabel("Aoristic Values", fontsize=15)

    ax.set_xlabel("24 Hours of the Day",fontsize=15)

    ax.set_title('Weighted Hours for '+chart_offense,fontsize=18)
#Grouping vehicle burglaries together

VehicleBurglary=crimes[crimes['OFFENSE_CATEGORY_ID']=='theft-from-motor-vehicle']

VehicleBurglary_line=VehicleBurglary.sum()



#applying the chart

Aoristic_Chart(VehicleBurglary_line, 'Vehicle Burglaries')
#Group accidents together

accidents=crimes[crimes['OFFENSE_CATEGORY_ID']=='traffic-accident']

accidents_line=accidents.sum()



#applying the chart

Aoristic_Chart(accidents_line,"Traffic Accidents")
#Getting my Vehicle Burglary data

VehicleBurglary_Clusters=VehicleBurglary[['GEO_LAT','GEO_LON']]
def Elbow_Method(k_clusters,data):

    errors = []

    K = range(1,k_clusters)

    for k in K:

        kmeanModel = KMeans(n_clusters=k).fit(data)

        kmeanModel.fit(data)

        errors.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

        

    plt.figure(figsize=(10,6))

    plt.plot(K,errors)

    plt.title('Error Rate vs. K Value')

    plt.xlabel('K')

    plt.ylabel('Error Rate')
Elbow_Method(40,VehicleBurglary_Clusters)
kmeans = KMeans(n_clusters=8)

kmeans.fit(VehicleBurglary_Clusters)

kmeans.cluster_centers_

VehicleBurglary['Clusters']=kmeans.labels_



sns.lmplot(x='GEO_LAT',y='GEO_LON',data=VehicleBurglary,

           hue='Clusters',

           fit_reg=False,

           size=10)
def Aoristic_And_Clusters(data,cluster_number):

    

    cluster=data[data['Clusters']==cluster_number]

    cluster_line=cluster.sum()

    

    

    fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(22,6))

    

    sns.regplot(x='GEO_LAT',y='GEO_LON',data=cluster,

           fit_reg=False,color='red',ax=ax1)

    

    ax2=cluster_line.iloc[21:45].plot(xticks=np.arange(0,24,1))
Aoristic_And_Clusters(VehicleBurglary,0)
Aoristic_And_Clusters(VehicleBurglary,1)
Aoristic_And_Clusters(VehicleBurglary,2)
Aoristic_And_Clusters(VehicleBurglary,3)
Aoristic_And_Clusters(VehicleBurglary,4)