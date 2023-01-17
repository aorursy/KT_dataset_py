import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_pluto = pd.read_csv('/kaggle/input/nyc-pluto/pluto_20v2.csv')
df= pd.read_csv('/kaggle/input/nyc-311-hpd-calls/311_Service_Requests_from_2010_to_Present.csv')
print('Shape of NYC 311 Dataframe is ',df.shape)
print('Shape of PLUTO Dataframe is ',df_pluto.shape)
df.head()
df.columns
df_pluto=df_pluto[['address','bldgarea','bldgdepth','builtfar','commfar','facilfar','lot','lotarea',
                   'lotdepth','numbldgs','numfloors','officearea','resarea','residfar','retailarea',
                   'yearbuilt','yearalter1','zipcode','ycoord','xcoord']]

df=df[['Unique Key', 'Created Date', 'Closed Date',
       'Complaint Type', 'Descriptor', 'Location Type', 'Incident Zip',
       'Incident Address', 'Street Name','Address Type',
       'City', 'Status', 'Due Date',
       'Resolution Description','Borough',
       'Latitude', 'Longitude']]


print('Shape of PLUTO dataframe is ',df_pluto.shape)
print('Shape of the NYC 311 call dataframe is', df.shape)

df.head()
df[['Address Type']].describe()
df=df.drop(columns=['Address Type'])
df_comp=df.groupby('Complaint Type')[["Unique Key"]].count()
df_comp=df_comp[df_comp['Unique Key']>80000].sort_values(by='Unique Key')
df_comp.columns=['No of complaints']


df_comp.plot(kind='barh',figsize=(10,8))
plt.xlabel('Number of Complains  $x10^6$', fontsize=14)
plt.ylabel('Complaint Type',fontsize=14)
plt.show()
print(df.shape)
df=df[df['Complaint Type'].isin(df_comp.index)]
print(df.shape)
df_bor= df.groupby('Borough')[['Unique Key']].count().sort_values('Unique Key',ascending=True)
df_bor.plot(kind='barh',figsize=(15,10))
plt.xlabel('Number of Complains  $x10^6$', fontsize=14)
plt.ylabel('Borough',fontsize=14)
plt.show()
df_zip=df.groupby('Incident Zip')[['Borough']].agg(lambda x:x.value_counts().index[0])
df_zip.head()
for i,j in zip(df[df['Borough']=='Unspecified'].index,df[df['Borough']=='Unspecified']['Incident Zip']):
    if np.isnan(j):
        continue
    df.at[i,'Borough']=df_zip.at[j,'Borough']
    #print(type(j))
    
df.groupby('Borough')[['Unique Key']].count().sort_values('Unique Key').plot(kind='barh',figsize=(10,8))
#!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS
print('Import Successfull')
from collections import Counter

count_dict = Counter(df['Incident Address'])
stopwords= set(STOPWORDS)
wc = WordCloud(background_color='white', max_words=20, stopwords=stopwords).generate_from_frequencies(count_dict)
#unique_string = ("").join(list(df['Incident Address'].astype('str')))
#wc = WordCloud(background_color='white', max_words=20, stopwords=stopwords).generate(unique_string)


plt.figure(figsize=(15,10))
plt.imshow(wc)
plt.axis('off')
plt.show()
df.groupby('Incident Address')[['Unique Key']].count().sort_values('Unique Key',ascending=False).head()
df[df['Incident Address']=='34 ARDEN STREET'][['Incident Address','Incident Zip','Borough']].head(1)

df.groupby('Incident Zip')[['Unique Key']].count().sort_values('Unique Key',ascending=False).head()
df.groupby('Status')[['Unique Key']].count().sort_values('Unique Key',ascending=False).head()
df_pluto['bldgage']=2020-df_pluto['yearbuilt']
df_pluto.head()
df[df['Complaint Type']=='HEAT/HOT WATER'].head()
df_comp_count=df[df['Complaint Type']=='HEAT/HOT WATER'].groupby('Incident Address')[['Incident Address']].count()
df_comp_count.columns=['count of complaints']
df_comp_count['address']=df_comp_count.index
df_comp_count.head()
#df_comp_count.index=None
df_comp_count.reset_index(drop=True,inplace=True)
df_comp_count.head()
df_corr = pd.merge(df_comp_count,df_pluto,on='address')
df_corr.head()
#df_corr['alterage']= 2020-df_corr['yearalter1']
df_corr.drop(columns=['yearbuilt'],inplace=True)
df_corr.head()
data = df_corr.drop(columns=['address'])

null_data=data.isnull()

for i in null_data.columns:
    result=0
    for j in null_data[i]:
        if j: result+=1
    print(i," has ",result, " null values")



data.describe()
for i in null_data.columns:
    result=0
    for k,j in enumerate(null_data[i]):
        if j: data.at[k,i]= data[i].mean()
            
            
null_data=data.isnull()
            
for i in null_data.columns:
    result=0
    for j in null_data[i]:
        if j: result+=1
    print(i," has ",result, " null values")
