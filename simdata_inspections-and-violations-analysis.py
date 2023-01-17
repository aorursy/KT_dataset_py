# import required libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import colorama
from colorama import Fore
from pandas import Series
from collections import Counter
# Load the dataset
df1 = pd.read_csv('../input/inspections.csv')
df2 = pd.read_csv('../input/violations.csv')
df1.head()
# all numerical columns
df1.describe()
# all categorical columns
categorical = df1.dtypes[df1.dtypes =="object"].index
df1[categorical].describe()
print(Fore.BLUE +'\n250 Inspectors inspected 39,283 different Facilities with 36,627 different Facilities_Names across LA County during time period 2015-2017.\n')
df1['facility_address'].value_counts().head(10)
print(Fore.BLUE +'\n 1000 VIN SCULLY AVE (Los Angeles) has appeared 383 times whose corresponding facility_name is DODGER STADIUM\n')
# What are the top 10 facility_cities for violations?
df1['facility_city'].value_counts().head(10)
# Unique facility_city
f_city=df1['facility_city'].nunique()
f_city_restaurant = df1['facility_city'].value_counts()[0]

#Plot Bar Graph
Bplot = df1['facility_city'].value_counts() 
Bplot[:11].plot(kind = 'barh', rot = 0)
print(Fore.BLUE +'\nOut of', f_city, 'cities, Los Angeles has total of',  df1['facility_city'].value_counts()[0], 'restaurants/facilities inspected during the time period of 2015-2017\n')
# TOPMOST FACILITY IN LOS ANGELES

df1[df1['facility_city']=='LOS ANGELES']['facility_name'].value_counts()[:11]
df1['facility_id'].value_counts().head(10)
# Name of facility with facility_id =FA0019271

top_fac_id = df1[df1['facility_id'] == 'FA0019271'] ['facility_name'].iloc[0]

print (Fore.BLUE +'\nSo,', top_fac_id, 'has facility_id == FA0019271.')
print (Fore.BLUE +'We have already explored DODGER STADIUM and here, it marks at top most facility_id as well.\n')
df1['facility_name'].value_counts().head()
fac_name = df1['facility_name'].value_counts()
fac_name[:20].plot(kind = 'barh', rot =0 )
fac_name1 = df1['facility_name'].value_counts()[0]
print (Fore.BLUE +'SUBWAY has appeared',fac_name1,'times.')
df1[df1['facility_name']=='SUBWAY']['facility_city'].value_counts()
# What are the top 5 facility_zip codes?
df1['facility_zip'].value_counts().head()
f_zip = df1[df1['facility_zip'] == '90012']['facility_city'].iloc[0]
df1['facility_zip'].value_counts().head(10).plot.bar()
print(Fore.BLUE+ 'The topmost facility_zip: 90012 is in',f_zip,)
df1['grade'].value_counts().plot.bar().colormap='Paired'
# facilities with POOR grade
df1[df1['grade']==' ']['facility_name']
sns.barplot(x='grade', y='score', data=df1, palette="rocket")
print(Fore.BLUE+'Above 32 Facilities received a Poor Grade/Score Card because they scored less than 70 in scores at various locations.(According to LA County Public Health: Poor in food handling practices and overall general food facility maintenance..)')
df1['owner_id'].value_counts().head(10)
df1[df1['owner_id']=='OW0029458']['owner_name'].iloc[0]
print(Fore.BLUE +df1[df1['owner_id']=='OW0029458']['owner_name'].iloc[0],'(OW0029458) has occurred', '{:,}'.format(df1['owner_id'].value_counts().head(10)[0]), 'times in the dataset.\n')
owner_name= df1['owner_name'].value_counts()
owner_name[:10].plot(kind='barh', rot=0)
df1['pe_description'].unique()
prog_name= df1['program_name'].value_counts()
prog_name[:10].plot(kind='barh')
sns.countplot(x='program_status', data= df1, palette = 'deep', edgecolor=sns.color_palette("dark", 7))
sns.countplot(x='service_description', data = df1, palette="rocket",edgecolor=sns.color_palette("dark", 7))
# Plot Graph
plt.figure(figsize=(10,8))
df1['score'].value_counts().sort_index().plot.bar()
sns.countplot(x='service_code', data = df1,palette="vlag",edgecolor=sns.color_palette("dark", 7))
plt.figure(figsize=(10,6))
sns.countplot(x='program_element_pe', data=df1, palette="rocket")
df1.hist(figsize=(10,10))
plt.show()
# correlation_matrix
corr_mat = df1.corr()
fig = plt.figure(figsize=(8,7))
sns.heatmap(corr_mat, vmax=0.9, square =True, annot=True)
plt.show()
df1[df1['score']>90].plot.hexbin(x='score', y='program_element_pe', gridsize=15)
df1['activity_date']=pd.to_datetime(df1['activity_date'])
df1['Year'] = df1['activity_date'].apply(lambda time: time.year)
df1['Month'] = df1['activity_date'].apply(lambda time: time.month)
df1['Day of Week'] = df1['activity_date'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thur', 4:'Fri', 5:'Sat', 6:'Sun'}
df1['Day of Week'] = df1['Day of Week'].map(dmap)
plt.figure(figsize=(10,8))
sns.countplot(x='Day of Week', data = df1, palette = 'rocket')
plt.figure(figsize=(10,8))
sns.countplot(x='Day of Week', data = df1, hue='program_status')
# type of inspection based on 'Day of Week'
plt.figure(figsize=(10,8))
sns.countplot(x='Day of Week', data = df1, hue='service_description', palette='viridis')
# Inspection based on 'Month'

plt.figure(figsize=(10,8))
sns.countplot(x='Month', data = df1, hue='service_description', palette='viridis')
byYear = df1.groupby('Year').count()
byYear['facility_zip'].plot.bar()
plt.figure(figsize=(13,7))
df1['Year1'] = df1["Year"].astype(str)
ax = plt.scatter("Year1","score", data=df1,
            c=df1["score"], cmap="inferno",
            s=900,alpha=.7,
            linewidth=2,edgecolor="k",)

#plt.colorbar()
plt.xticks(df1["Year1"].unique())
plt.yticks(np.arange(60,200,20))
plt.title('Health Violations over the years',color='b')
plt.xlabel("year")
plt.ylabel("score")
plt.show()
plt.figure(figsize=(20,15))
df1.groupby('activity_date').count()['serial_number'].plot()

plt.title('Health Code Violations over Time', color='b')
plt.show()
df2.head()
df2['serial_number'].describe()
# Verify serial number to corresponding violation description
df2[df2['serial_number'] == 'DAT2HKIRE']['violation_description']
# Verify violation code to corresponding serial number

df2[df2['violation_code'] == 'F033']['serial_number']
print(Fore.BLUE+'It is clear that a serial_number may refer to more than one violations and in the same manner, a violation_code also refers to more than one serial_number.')
# Most common health code violations with violation_description
topd = df2['violation_description'].value_counts().head(30)

topd[:20].plot(kind='barh', rot=0)
# merging both DataFrames using Right Join

df3=pd.merge(df1, df2, on ='serial_number', how = 'right')
# object 'subway'  created which contains corresponding violation description information

subway=df3[df3['facility_name'] == 'SUBWAY']['violation_description']
# DataFrame 'df_new' created 

df_new = pd.DataFrame(subway)
df_new.head()
# DataFrame 'df_new1' created which holds the count of 40 unique violations at Subway in new column 'Count' 

df_new1=df_new.groupby(['violation_description']).size().reset_index().sort_values(0, ascending=False)
df_new1.rename(columns = {0:'Count'}, inplace = True)
df_new1[:20]