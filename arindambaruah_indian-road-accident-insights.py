import os
import numpy as np
import pandas as pd
import plotly as py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import contextily as ctx
import matplotlib.image as mpimg
import matplotlib
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs 
from plotly.offline import init_notebook_mode
from plotly.offline import plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns
import mpl_toolkits


data=pd.read_csv('../input/road-accidents-in-india/only_road_accidents_data3.csv')
data.head()
data_by_states=data.loc[:,['STATE/UT','YEAR','Total']]
data_by_states.head()
df1=data_by_states.drop('YEAR',axis=1)
df1.head()
df1=df1.groupby('STATE/UT').sum().reset_index().sort_values(by='Total',ascending=False)
df1.head()
plt.figure(figsize=(20,10))

sns.barplot(x=df1['STATE/UT'],y=df1['Total'],palette='rainbow')
plt.xticks(rotation=90)

plt.title('Cases of road accidents in each state/UT from 2001-14',size=20)

df_top=df1[df1['Total']>300000]
df_top
plt.figure(figsize=(20,10))
sns.barplot(x=df_top['STATE/UT'],y=df_top['Total'],palette='gnuplot_d')
plt.xticks(size=15,rotation=45)
plt.yticks(size=10)

plt.title('States with highest road accidents between 2001-14',size=20)
df_year=data_by_states.groupby('YEAR').sum().reset_index()
df_year
df_year['Percentage change']=df_year['Total'].pct_change().round(2)
df_year
plt.figure(figsize=(20,10))

plt.plot(df_year['YEAR'],df_year['Total'], color='indianred')
plt.title('Number of road accidents in India on a yearly basis',size=20)
plt.xlabel('Year',size=15)
plt.ylabel('Number of accidents',size=15)
plt.xticks(np.arange(2001,2015,1),size=10)
plt.figure(figsize=(20,10))
sns.barplot(x=df_year['YEAR'],y=df_year['Percentage change'],palette='viridis')
plt.xlabel('Year',size=15)
plt.ylabel('Perncentage Change',size=15)
plt.yticks(np.arange(-0.02,0.09,0.01))
plt.figure(figsize=(20,10))

plt.plot(df_year['YEAR'],df_year['Total'], color='indianred')
plt.title('Number of road accidents in India on a yearly basis',size=20)
plt.xlabel('Year',size=15)
plt.ylabel('Number of accidents',size=15)
plt.xticks(np.arange(2001,2015,1),size=10)
from sklearn.linear_model import LinearRegression

x_matrix=df_year['YEAR'].values.reshape(-1,1)
y=df_year['Total'].values
reg=LinearRegression()
reg.fit(x_matrix,y)
reg.intercept_
reg.coef_
regression=[]
a=np.arange(2015,2025)
for year in a:
    year=np.array([year])
    year=year.reshape(1,-1)
    regression.append(reg.predict(year))

df_predicted=pd.DataFrame(regression)
df_predicted=df_predicted.rename(columns={0:'Total'})
df_predicted.reset_index(drop=True,inplace=True)
years=np.arange(2015,2025,1)
df_temp=pd.DataFrame(years)
df_temp
df_predicted['Year']=df_temp
df_predicted=df_predicted[['Year','Total']]
df_predicted['Total']=df_predicted['Total'].values.round(0)
df_predicted.index=df_predicted.index+1
df_predicted
df_location=pd.read_csv('../input/indian-road-accidents-data/No_of_Road_Acc_acco_to_clf_of_age_of_Driver_2014_2016.csv')
df_location.head()
df_location_updated=df_location.iloc[:,47:]
df_location_updated.head()

df_month=pd.read_csv('../input/road-accidents-in-india/only_road_accidents_data_month2.csv')
df_month.head()
i=0
for year in df_month['YEAR']:
    if year==2014:
        i+=1
    else:
        df_month=df_month.drop(i)
        i+=1
df_month.reset_index(drop=True)
df_month_top=df_month[df_month['TOTAL']>20000]
df_month_top
plt.figure(figsize=(20,10))
sns.barplot(x='STATE/UT',y='TOTAL',data=df_month_top,palette='winter')
plt.xlabel('State/UT',size=15)
plt.ylabel('Total accidents',size=15)
plt.title('States with highest accidents in 2014',size=25)
df_tamil_nadu=df_month.loc[df_month['STATE/UT'].isin(['Tamil Nadu'])]
df_karnataka=df_month.loc[df_month['STATE/UT'].isin(['Karnataka'])]
df_maharashtra=df_month.loc[df_month['STATE/UT'].isin(['Maharashtra'])]
df_delhi=df_month.loc[df_month['STATE/UT'].isin(['Delhi Ut'])]
df_tamil_nadu
df_delhi
df_karnataka
df_maharashtra
plt.figure(figsize=(20,10))
val=df_maharashtra.iloc[:,1:14].values
plt.plot(val[0],color='indianred',linewidth=3)
plt.xlim(1,13)
plt.ylim(3000,4500)
plt.xlabel('Month',size=15)
plt.ylabel('Number of accidents',size=15)
plt.title('Number of accidents on monthly basis in Maharshtra (2014)',size=25)
plt.xticks(np.arange(1,13))
plt.figure(figsize=(20,10))
val=df_karnataka.iloc[:,1:14].values
plt.plot(val[0],color='green',linewidth=3)
plt.xlim(1,13)
plt.ylim(3000,4500)
plt.xlabel('Month',size=15)
plt.ylabel('Number of accidents',size=15)
plt.title('Number of accidents on monthly basis in Karnataka (2014)',size=25)
plt.xticks(np.arange(1,13))
plt.figure(figsize=(20,10))
val=df_tamil_nadu.iloc[:,1:14].values
plt.plot(val[0],color='purple',linewidth=3)
plt.xlim(1,13)
plt.ylim(5000,6000)
plt.xticks(np.arange(1,13))
plt.xlabel('Month',size=15)
plt.ylabel('Number of accidents',size=15)
plt.title('Number of accidents on monthly basis in Tamil Nadu (2014)',size=25)

plt.figure(figsize=(20,10))
val=df_delhi.iloc[:,1:14].values
plt.plot(val[0],color='black',linewidth=3)
plt.xlim(1,13)
plt.ylim(600,850)
plt.xlabel('Month',size=15)
plt.ylabel('Number of accidents',size=15)
plt.title('Number of accidents on monthly basis in New Delhi (2014)',size=25)
plt.xticks(np.arange(1,13))
df_hedu=pd.read_csv('../input/indian-road-accidents-data/Accidents_Classified_Acc_To_EduQual_Of_Drivers_9-10_standard-09-16.csv')
df_hedu
df_ledu=pd.read_csv('../input/indian-road-accidents-data/Accidents_Acc_EduQual_Drivers_above_10_Standard_09-16.csv')
df_ledu.head()
df_hedu_updated=df_hedu.loc[:,['States/UTs','2016']]
df_hedu_updated=df_hedu_updated.rename(columns={'2016':'10th pass'})
df_hedu_updated
df_ledu_updated=df_ledu.loc[:,['2016']]
df_ledu_updated=df_ledu_updated.rename(columns={'2016': '10th Fail'})
df_ledu_updated
df_education=df_hedu_updated.copy()
df_education['10th fail']=df_ledu_updated
df_education.dropna(axis=0,inplace=True)
df_education.drop(36,inplace=True)
plt.figure(figsize=(20,10))
sns.barplot(x=df_education['States/UTs'],y=df_education['10th pass'],palette='Paired')
plt.xticks(rotation=45,size=10)
plt.title('Drivers involved in road accidents who are 10th pass or above (2016)',size=25)
plt.ylabel('Number of accidents',size=15)
plt.xlabel('State/UT',size=15)
plt.figure(figsize=(20,10))
sns.barplot(x=df_education['States/UTs'],y=df_education['10th fail'])
plt.xticks(rotation=45,size=10)
plt.title('Drivers involved in road accidents who are not 10th pass (2016)',size=25)
plt.ylabel('Number of accidents',size=15)
plt.xlabel('State/UT',size=15)
df_education['Total accidents']=df_education['10th pass']+df_education['10th fail']
df_education.head()
df_education=df_education.sort_values(by='Total accidents',ascending=False)
df_age=pd.read_csv('../input/indian-road-accidents-data/No_of_Road_Acc_acco_to_clf_of_age_of_Driver_2014_2016.csv')
df_age.head()
df_age_2014=df_age.iloc[:,0:6]
df_age_2014.drop('S. No.',axis=1,inplace=True)
#df_age_2014.drop(36,inplace=True)
df_age_2014.dropna(axis=1,inplace=True)
plt.figure(figsize=(20,10))
sns.barplot(x='State/ UT',y='0-14',data=df_age_2014)
plt.xticks(rotation=90)
plt.title('Drivers who are below the age of 14 and involved in accidents (2014)',size=25)
plt.xlabel('State/UTs',size=15)
plt.ylabel('Number of accidents',size=15)
plt.figure(figsize=(20,10))
sns.barplot(x='State/ UT',y='15-24',data=df_age_2014,palette='gnuplot')
plt.xticks(rotation=90)
plt.title('Drivers who are of ages 15-24 and involved in accidents (2014)',size=25)
plt.xlabel('State/UTs',size=15)
plt.ylabel('Number of accidents',size=15)
plt.figure(figsize=(20,10))
sns.barplot(x='State/ UT',y='25-64',data=df_age_2014)
plt.xticks(rotation=90)
plt.title('Drivers who are of ages 25-64 and involved in accidents (2014)',size=25)
plt.xlabel('State/UTs',size=15)
plt.ylabel('Number of accidents',size=15)
plt.figure(figsize=(20,10))
sns.barplot(x='State/ UT',y='64 above',data=df_age_2014,palette='Spectral')
plt.xticks(rotation=90)
plt.title('Drivers who are of ages 64 above and involved in accidents (2014)',size=25)
plt.xlabel('State/UTs',size=15)
plt.ylabel('Number of accidents',size=15)
df_Andhra=df_age_2014.iloc[0:1,:]
df_Andhra
unstack_df=df_Andhra.unstack()
unstack_df.reset_index(drop=True,inplace=True)
list1=[]
for i in range(1,5):
    list1.append(unstack_df[i])
    i+=1
    
labels=['0-14','15-24','25-64','64 above']
plt.figure(figsize=(10,10))
plt.pie(list1,labels=labels,autopct='%1.1f%%')
plt.title('Road accidents in Andhra Pradesh by age groups (2014)',size=20)

df_maha=df_age_2014.loc[14:14,:]
df_maha
unstack_df=df_maha.unstack()

unstack_df.reset_index(drop=True,inplace=True)

list1=[]
for i in range(1,5):
    list1.append(unstack_df[i])
    i+=1
plt.figure(figsize=(10,10))
plt.pie(list1,labels=labels,autopct='%1.1f%%')
plt.title('Road accidents in Maharashtra by age groups (2014)',size=20)
df_Kar=df_age_2014.copy()

df_Kar=df_Kar.iloc[11:12,:]
df_Kar

unstack_df=df_Kar.unstack()
unstack_df.reset_index(drop=True,inplace=True)
list1=[]
for i in range(1,5):
    list1.append(unstack_df[i])
    i+=1
plt.figure(figsize=(10,10))
plt.pie(list1,labels=labels,autopct='%1.1f%%')
plt.title('Road accidents in Karnataka by age groups (2014)',size=20)
df_time=pd.read_csv('../input/road-accidents-in-india/only_road_accidents_data3.csv')
df_time=df_time.reset_index(drop=True)
df_time.head()

df_time=df_time.loc[df_time['YEAR'].isin(['2014'])]
df_time
df_time_top=df_time[df_time['Total']>40000]
df_time_top.reset_index(drop=True,inplace=True)
df_time_Andhra=df_time_top.loc[0:0,:]
df_time_Andhra

unstack_df=df_time_Andhra.unstack()
unstack_df.reset_index(drop=True,inplace=True)
unstack_df
labels=['0-3','3-6','6-9','9-12','12-15','15-18','18-21','21-24']
list1=[]
for i in range(2,10):
    list1.append(unstack_df[i])
    i+=1
plt.figure(figsize=(10,10))
plt.pie(list1,labels=labels,autopct='%1.1f%%')
plt.title('Road accidents in Andhra Pradesh by time of accident in hrs (2014)',size=20)
df_time_Kar=df_time_top.loc[1:1,:]
df_time_Kar

unstack_df=df_time_Kar.unstack()
unstack_df.reset_index(drop=True,inplace=True)
unstack_df
list1=[]
for i in range(2,10):
    list1.append(unstack_df[i])
    i+=1
plt.figure(figsize=(10,10))
plt.pie(list1,labels=labels,autopct='%1.1f%%')
plt.title('Road accidents in Karanataka by time of accident in hrs (2014)',size=20)
df_time_TN=df_time_top.loc[3:3,:]
df_time_TN
unstack_df=df_time_TN.unstack()
unstack_df.reset_index(drop=True,inplace=True)
unstack_df
list1=[]
for i in range(2,10):
    list1.append(unstack_df[i])
    i+=1
plt.figure(figsize=(10,10))
plt.pie(list1,labels=labels,autopct='%1.1f%%')
plt.title('Road accidents in Tamil Nadu by time of accident in hrs (2014)',size=20)
