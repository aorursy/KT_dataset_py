#importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Read the csv file
df1 = pd.read_csv('../input/311_Service_Requests_from_2010_to_Present.csv',index_col = 0)
df1.shape
#first 5 columns
df1.head(5)
#df1.columns
#Info of the DF
df1.info()
df1.describe(include = 'all')
#count of non- nulls in each column
df1[df1.columns].count()
#Checking for Missing values
plt.figure(figsize=(10,6))
sns.heatmap(df1.isnull(), cbar=False)
plt.show() #white indicates null values
#There are few columns with full blanks and few columns with most blanks
#Count of nulls in each column
df1.isnull().sum(axis = 0)
#removing the unwanted columns
df1.columns
remove_cols = ['Agency', 'Cross Street 1', 'Cross Street 2',
       'Intersection Street 1', 'Intersection Street 2',
        'Landmark', 'Facility Type', 'X Coordinate (State Plane)',
       'Y Coordinate (State Plane)', 'Park Facility Name', 
       'School Name', 'School Number', 'School Region', 'School Code',
       'School Phone Number', 'School Address', 'School City', 'School State',
       'School Zip', 'School Not Found', 'School or Citywide Complaint',
       'Vehicle Type', 'Taxi Company Borough', 'Taxi Pick Up Location',
       'Bridge Highway Name', 'Bridge Highway Direction', 'Road Ramp',
       'Bridge Highway Segment', 'Garage Lot Name', 'Ferry Direction',
       'Ferry Terminal Name','Location']
df1.drop(remove_cols, axis=1, inplace= True)
df1.columns
df1.shape
df1[df1.columns].count()
#remove rows with missing data of latitiude and long.
df1 = df1[(df1['Latitude'].notnull()) & (df1['Longitude'].notnull())]
df1[df1.columns].count()
df1['Complaint Type'].nunique()
#remove the nulls in Incident zip and city too
df1 = df1[(df1['Incident Zip'].notnull()) & (df1['City'].notnull())]
df1[df1.columns].count()
#Request_Closing_Time
df1['Created Date'] = pd.to_datetime(df1['Created Date'])
df1['Closed Date'] = pd.to_datetime(df1['Closed Date'])
df1['Request_Closing_Time'] = df1['Closed Date']-df1['Created Date'] 
df1['Request_Closing_Time'].describe()
df1['Created_Hour'] = df1['Created Date'].apply(lambda x: x.hour)
df1['Created_Month'] = df1['Created Date'].apply(lambda x: x.month)
df1['Created_Day of Week'] = df1['Created Date'].apply(lambda x: x.dayofweek)
df1.head(5)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
df1['Created_Day of Week']=df1['Created_Day of Week'].map(dmap)
df1.head(5)
#plotting the count based on day of the week
plt.figure(figsize=(10,6))
sns.countplot(x='Created_Day of Week', data = df1, order=df1['Created_Day of Week'].value_counts().index, palette ="Reds_d")
#more number of complaints on Sat and Sun
#plotting the count based on Month
plt.figure(figsize=(10,6))
sns.countplot(x='Created_Month', data = df1, palette ="Blues_d")
#No complaints on Jan, Feb months and very less complaints on March
#the above plot has missing months. So, we will use line chart to show the trend
bymonth = df1.groupby('Created_Month').count()
#taking the latitude as reference
plt.figure(figsize=(8,5))
bymonth['Latitude'].plot()
#No complaints on Jan, Feb months and very less complaints on March
#plotting the count based on Hour
plt.figure(figsize=(8,5))
sns.countplot(x='Created_Hour', data = df1)
#Frequency of complaints are more during late hours (10 and 11'o CLock)
#top 5 zipcodes that call 911
df1['Incident Zip'].value_counts().head(5)
#Boroughs that call 911
df1['Borough'].value_counts().head(5).plot(kind='bar', figsize=(6,4), title=('Complaints by Borough'))
#complaint categories that call 911 in descending order
df1['Complaint Type'].value_counts().sort_values().plot(kind = 'barh',figsize=(12,10), title=('Complaint Counts'), color = 'Orange')

#Stacked Horizontal Bar Chart based on different Complaint Types from each borough
CT_borough = df1.groupby(['Complaint Type','Borough']).size().unstack()
CT_borough = CT_borough.sort_values('BROOKLYN', axis = 0,ascending=True)
CT_borough.plot(kind='barh', title='Different Complaints stacked by each Borough', figsize=(20,12), stacked = True);
#Count plots on Complaint type
CT_borough = df1.groupby(['Complaint Type','Borough']).size().unstack()
CT_borough.sort_values('Complaint Type').plot(kind='bar', title='Each Complint Type distribution from Different Boroughs', figsize=(20,5));
#Top 10 Complaints with most CTypes per borough
col_number = 2
row_number = 3
fig, axes = plt.subplots(row_number,col_number, figsize=(12,12))

for i, (label,col) in enumerate(CT_borough.iteritems()):
    ax = axes[int(i/col_number), i%col_number]
    col = col.sort_values(ascending=False)[:10]
    col.sort_values().plot(kind='barh', ax=ax,color = 'orange')
    ax.set_title(label)
    
plt.tight_layout()
df1['Request_Closing_Time_Float'] = df1['Request_Closing_Time'].apply(lambda x:x/np.timedelta64(1, 'h'))
gh= df1.groupby(['Created_Month','Borough'])
gh['Request_Closing_Time_Float'].mean().unstack().plot(figsize=(15,7),title='Processing time per Borough')
avg_closing = df1.groupby(['Complaint Type','Borough'])
avg_closing['Request_Closing_Time_Float'].mean().unstack()
df1['Complaint Type'].value_counts()
df1['Highlevel_CT'] =["Noise" if CT == 'Noise - Street/Sidewalk' or CT=='Noise - Commercial' or CT=='Noise - Vehicle' or CT =='Noise - Park' or CT =='Noise - House of Worship' else CT for CT in df1["Complaint Type"]]
df1['Highlevel_CT'].value_counts()[:3]
df1['Request_Closing_Time'].fillna(df1['Request_Closing_Time'].mean(),inplace=True)
df1['Request_Closing_Time_sec']=df1['Request_Closing_Time'].dt.total_seconds()
df1.groupby(['Highlevel_CT']).agg(['mean'])[['Request_Closing_Time_sec']]

#Top3=df1.Highlevel_CT.value_counts()[:3].index 
HCT_Closing =df1.loc[:,['Highlevel_CT','Request_Closing_Time_sec']]
HCT_Closing
#df1.Highlevel_CT.isin(Top3)
grand_mean=HCT_Closing.Request_Closing_Time_sec.mean() 
grand_mean
group_mean=HCT_Closing.groupby('Highlevel_CT').mean()['Request_Closing_Time_sec'] 
group_mean
gn = ['low','med','high']
HCT_Closing['group'] = pd.cut(HCT_Closing['Request_Closing_Time_sec'], bins=3, labels=gn) 
HCT_Closing['group'].unique()
contigency_table = pd.crosstab(HCT_Closing.Highlevel_CT,HCT_Closing.group)
print(contigency_table)
import scipy.stats as stats
chi_square , p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(contigency_table)
chi_square, '{0:f}'.format(p_value)
df1['Location Type'].unique()
CT_Loc = df1.loc[:,['Highlevel_CT','Location Type']]
CT_Loc.dropna(inplace=True)
CT_Loc
contigency_table1 = pd.crosstab(CT_Loc.Highlevel_CT,CT_Loc['Location Type'])
print(contigency_table1)
contigency_table1.head()
chi_square , p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(contigency_table1)
chi_square, '{0:f}'.format(p_value)
