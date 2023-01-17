import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_zomato=pd.read_csv('ZomatoRestaurantsIndia.csv')
df_zomato.head(3)
print(df_zomato.shape)
df_zomato.dtypes
df_zomato.head()
#First drop all the duplicate rows
df=df_zomato.copy()  # Copy the original dataframe

df1=df_zomato.drop_duplicates()

print(df1.shape)

#Columns currency , country_id have same value for all rows,they can be dropped.Column url does not give much information and locality_verbose is a repetitive column.

df1.drop(['url','currency','country_id','locality_verbose','opentable_support','address'],axis=1,inplace=True)
print(df1.shape)
print(df1.dtypes)
df1[['res_id','delivery','takeaway']]=df1[['res_id','delivery','takeaway']].astype('object')
df1.dtypes
dup_len=df1[df1.duplicated(['res_id'],keep=False)]
print(len(dup_len))

df1.drop_duplicates(subset='res_id',keep=False,inplace=True)
df1.shape
name_count=df1['name'].value_counts()
print(len(name_count))

df1['name'].value_counts()[:10].plot(kind='bar')
plt.title("Top restaurant chain in Country",weight="bold")
plt.ylabel("Name")
plt.xlabel("Count")
plt.show()
estab_count=df1['establishment'].unique()
print(len(estab_count))

#var=df1.groupby(['establishment']).count()
#print(var.head(2))

print("Mode is ",df1['establishment'].mode())
sns.countplot(df1['establishment'])
plt.title("Count of Type of Restaurant",weight='bold')
plt.xticks(rotation=90)

plt.show()
city_count=df1['city'].unique()
print("No. of differen cities are ",len(city_count))

city_mode=df1['city'].mode()
print("Most no. of restaurants are in ",city_mode)

print(df1['city'].isnull().sum())

fig,ax=plt.subplots(figsize=(18,6))
sns.countplot(df1['city'],ax=ax)
plt.xticks(rotation=90)
plt.show()
#Visualization
fig,ax=plt.subplots(figsize=(14,18))
df1.city.value_counts().plot(kind='barh',ax=ax)
plt.title("Restaurant count in city",weight="bold")
plt.xlabel("Count")
plt.ylabel("City")
plt.show()

#Showing Numbers
df1.groupby('city')['name'].count().sort_values()
print("Total diff. types of locality are:",len(df['locality'].unique()),"\n")

df1.groupby('city')['locality'].value_counts()[:10].plot(kind='pie')
var=df1.groupby('city')['locality'].value_counts()
print('Count of localities in diff. city:\n',var)
print(df1['latitude'].dtype)
#(df1['latitude']<8.4) 
invalid_loc=(df1['latitude']<8.4) & (df1['latitude']>37.6)

df1['latitude']=df1['latitude'].replace(invalid_loc,np.nan)
#Find length of such location
print(len(invalid_loc))

df1['latitude'].isnull().sum()
print(df1['longitude'].dtype)
 
invalid_loc=(df1['longitude']<68.7) & (df1['longitude']>97.25)

df1['longitude']=df1['longitude'].replace(invalid_loc,np.nan)

print(len(invalid_loc))
df1['longitude'].isnull().sum()
df1['cuisines'].count().sum()

def cuisine_func(data):
    features=[]
    for i in data.cuisines:
        for j in i.split(','):
            j=j.strip()
            features.append(j)
    return(features)
data=df1[df1.cuisines.notnull()]
data.highlights=data.cuisines.apply(lambda x:x.lower().strip())
cuisine=cuisine_func(data)
cuisine=pd.Series(cuisine)
cuisine.value_counts()[:10]
print(df1['average_cost_for_two'].value_counts()[:10])
#df1['average_cost_for_two'].isnull().sum()
plt.figure(figsize = (12,8))
df1['average_cost_for_two'].value_counts()[:10].plot(kind = 'pie')
plt.title('Avg cost in Restaurent for 2 people', weight = 'bold')
plt.show()
print(df1['price_range'].value_counts())

print(df1['price_range'].isnull().values.any())

df1['price_range'].plot(kind='hist')
plt.show()
plt.figure(figsize = (12,6))
names = df1['price_range'].value_counts().index
values = df1['price_range'].value_counts().values
explode = (0.1, 0.1, 0.1, 0.1)  # explode 1st slice

colors = ['blue','red','green','yellow','blck','white']
plt.title('Price Range Restaurants', weight = 'bold')
plt.pie(values, explode=explode, labels=names, colors=colors,autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()
df1['highlights'].value_counts()[:10]
def highlight_func(data):
    features=[]
    for i in data.highlights:
        for j in i.split(','):
            j=j.strip()
            features.append(j)
    return(features)
data=df1[df1.highlights.notnull()]
data.highlights=data.highlights.apply(lambda x:x.lower().strip())
features=highlight_func(data)
plt.figure(figsize=(12,6)) 
pd.Series(features).value_counts()[:10].plot(kind='bar',color= 'c')
plt.title('Highlights of restaurant',weight='bold')
plt.xlabel('Highlights')
plt.ylabel('Count')
average_ratings=df1.groupby(['city'],as_index=False)
avg_agg=average_ratings['aggregate_rating'].agg(np.mean)
print(avg_agg.head())
plt.figure(figsize=(25,10))
plt.xlabel('City', fontsize=20)
plt.ylabel('Average Ratings', fontsize=20)
plt.title('Average Ratings on Cities', fontsize=30)
plt.bar(avg_agg['city'], avg_agg['aggregate_rating'])
plt.xticks(rotation=90)
plt.show()
df1['rating_text'].value_counts()
vote_df=df1.groupby(['city'],as_index=False)['votes'].mean()
print(vote_df.head(10))
vote_df.columns=['City','Mean Votes']
vote10=(vote_df.sort_values(['Mean Votes'],ascending=False)).head(10)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x="City",y="Mean Votes",data=vote10)
plt.xlabel('City')
plt.ylabel('Mean Votes')
plt.title('City vs Votes',weight='bold')
plt.show()
print('National mean of photos',df1['photo_count'].mean())
photo_df=df1.groupby(['city'],as_index=False)['photo_count'].mean()
photo_df.columns=['City','Mean Photo Count']
photo10=(photo_df.sort_values(['Mean Photo Count'],ascending=False)).head(10)
print(photo10.head(10))
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x="City",y="Mean Photo Count",data=photo10)
plt.xlabel('City')
plt.ylabel('Mean Photo Count')
plt.title("Photo Count of Cities",weight='bold')
plt.show()

border=[0.1,0.2,0.0]
fig,ax=plt.subplots()

df1['delivery'].value_counts().plot(kind='pie',explode=border,autopct="%1.1f%%",ax=ax)
plt.title("Delivery Count for Country Restaurant",weight='bold')
plt.show()
del_df=df1.groupby(['city'],as_index=False)['delivery'].count()
del_df.columns=['City','Delivery']
del10=(del_df.sort_values(['Delivery'],ascending=False)).head(10)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x="City",y="Delivery",data=del10)
plt.xlabel('City')
plt.ylabel('Delivery Count')
plt.title("Delivery Count of Cities",weight='bold')
plt.show()
print(del_df.head(5))
print(df1.isnull().values.any())
df1.dropna(inplace=True)
print(df1.info())
df1.describe()
df1.corr()
#Detecting by IQR
Q1=df1.quantile(0.25)
Q3=df1.quantile(0.75)
IQR=Q3-Q1
print(IQR)

#sns.boxplot(x='city_id',y='name',data=df1)
#plt.show()
def splitDataFrameIntoSmaller(df1, chunkSize = 1019): 
    listOfDf = list()
    numberChunks = len(df1) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df1[i*chunkSize:(i+1)*chunkSize])
    return listOfDf


df_list=splitDataFrameIntoSmaller(df1)
df_list[0].head()
df_list[0]['region']='north'
df_list[1]['region']='south'
df_list[2]['region']='west'
df_list[3]['region']='central'
df_list[4]['region']='east'

region_df=pd.DataFrame()

region_df=region_df.append(df_list)
region_df.head(5)

group_city=region_df.groupby(['region'],as_index=False)['city'].count()
group_city.columns=['Region','City']
print(group_city)
df_exp=df1[df1['price_range']>3]
df_exp['city'].value_counts()[:10].plot(kind='barh')
plt.xlabel("Count of restaurant")
plt.ylabel('City')
plt.title('Cities with expensive restaurant')
plt.show()
region_df.groupby('region').describe()
def highlight_func(data):
    features=[]
    for i in data.highlights:
        for j in i.split(','):
            j=j.strip()
            features.append(j)
    return(features)
feature=list()
data=df_list[0][df_list[0].highlights.notnull()]
data.highlights=data.highlights.apply(lambda x:x.lower().strip())
f1=highlight_func(data)
feature.extend(set(f1))

data=df_list[1][df_list[1].highlights.notnull()]
data.highlights=data.highlights.apply(lambda x:x.lower().strip())
f2=highlight_func(data)
feature.extend(set(f2))

data=df_list[2][df_list[2].highlights.notnull()]
data.highlights=data.highlights.apply(lambda x:x.lower().strip())
f3=highlight_func(data)
feature.extend(set(f3))

data=df_list[3][df_list[3].highlights.notnull()]
data.highlights=data.highlights.apply(lambda x:x.lower().strip())
f4=highlight_func(data)
feature.extend(set(f4))

data=df_list[4][df_list[4].highlights.notnull()]
data.highlights=data.highlights.apply(lambda x:x.lower().strip())
f5=highlight_func(data)
feature.extend(set(f5))
north=list(set(f1)-(set(f2)-set(f3)-set(f4)-set(f5)))
east=list(set(f5)-(set(f2)-set(f3)-set(f4)-set(f1)))
south=list(set(f2)-(set(f5)-set(f3)-set(f4)-set(f1)))
west=list(set(f3)-(set(f5)-set(f2)-set(f4)-set(f1)))
comm=list()
comm.extend(north)
comm.extend(south)
comm.extend(west)
comm.extend(east)

c=pd.Series(comm)
c.column=['High']
c.value_counts()[:10].plot(kind='bar')

plt.xlabel('Locality')
plt.ylabel('Count of highlight')
plt.title("Top 10 Highlights ",weight='bold')
#plt.show()
def cuisine_func(data):
    features=[]
    for i in data.cuisines:
        for j in i.split(','):
            j=j.strip()
            features.append(j)
    return(features)
feature=list()
data=df_list[0][df_list[0].cuisines.notnull()]
data.cuisines=data.cuisines.apply(lambda x:x.lower().strip())
f1=cuisine_func(data)
feature.extend(set(f1))

data=df_list[1][df_list[1].cuisines.notnull()]
data.cuisines=data.cuisines.apply(lambda x:x.lower().strip())
f2=cuisine_func(data)
feature.extend(set(f2))

data=df_list[2][df_list[2].cuisines.notnull()]
data.cuisines=data.cuisines.apply(lambda x:x.lower().strip())
f3=cuisine_func(data)
feature.extend(set(f3))

data=df_list[3][df_list[3].cuisines.notnull()]
data.cuisines=data.cuisines.apply(lambda x:x.lower().strip())
f4=cuisine_func(data)
feature.extend(set(f4))

data=df_list[4][df_list[4].cuisines.notnull()]
data.cuisines=data.cuisines.apply(lambda x:x.lower().strip())
f5=cuisine_func(data)
feature.extend(set(f5))
print(set(f1))
print(set(f5))
print(set(f2))
print(set(f3))
common_cuisine=list()

common_cuisine.extend(f1)
common_cuisine.extend(f2)
common_cuisine.extend(f3)
common_cuisine.extend(f5)

comm=pd.Series(common_cuisine)
comm.columns=['Cuisine']
comm.value_counts()[:10].plot(kind='bar')

plt.xlabel('Cuisine')
plt.ylabel('Number')
plt.title('Top 10 cuisines across region',weight='bold')


cuisine_count = []

for i in df_list[0].cuisines:
    for t in i.split(','):
        t = t.strip()
        cuisine_count.append(t)


plt.figure(figsize=(12,6))
pd.Series(cuisine_count).value_counts()[:10].plot(kind='bar',color= 'c')
plt.title('Top 10 cuisines in North Area',weight='bold')
plt.xlabel('Cuisine')
plt.ylabel('Count')
plt.show()
plt.figure(figsize = (12,6))
sns.barplot(x='aggregate_rating',y='photo_count',data=df_list[0])
plt.xlabel("Rating")
plt.ylabel("Photo Count")
plt.title("Rating Trend with Photo",weight = 'bold')
plt.show()
plt.figure(figsize = (12,6))
sns.barplot(x='aggregate_rating',y='votes',data=df_list[0])
plt.xlabel("Rating")
plt.ylabel("Votes")
plt.title("Rating Trend with Votes",weight = 'bold')
plt.show()
plt.figure(figsize = (12,6))
sns.boxplot(x='aggregate_rating',y='votes',data=df_list[0])
plt.xlabel("Rating")
plt.ylabel("Votes")
plt.title("Rating Trend with Votes",weight = 'bold')
plt.show()
new_df=region_df[region_df['city']=='Mumbai']
expensive=new_df[new_df['average_cost_for_two']>5000]


new=new_df.groupby(['name'],as_index=False)['average_cost_for_two'].sum()
new.columns=['Name',"Cost"]
costly=(new.sort_values(['Cost'],ascending=False)).head(20)
print(costly.head())

sns.barplot(y='Cost',x='Name',data=costly)
plt.xlabel("Reataurant Name")
plt.ylabel("Avg.Cost")
plt.title("Costly Restaurant in Mumbai",weight="bold")
plt.xticks(rotation=90)
plt.show()
cuisines= []

for i in new_df.cuisines:
    for j in i.split(','):
        j = j.strip()
        cuisines.append(j)
        
        
plt.figure(figsize=(12,6)) 
pd.Series(cuisines).value_counts()[:20].plot(kind='bar',color= 'r')
plt.title('Top 10 cuisines in Mumbai',weight='bold')
plt.xlabel('cuisines type')
plt.ylabel('No of restaurants')
plt.show()
new_df['locality'].value_counts()

# Checking Popularity of a locality by calculating average rating of all restaurants present in that locality.
pop_loc=new_df.groupby(['locality'],as_index=False)['aggregate_rating'].mean()
pop_loc.columns=['Locality','Rating']
loc_df=(pop_loc.sort_values(['Rating'],ascending=False)).head(10)

sns.barplot(x='Locality',y='Rating',data=loc_df)
plt.xticks(rotation=90)
plt.title("Popular Locality in Mumbai")
plt.show()
plt.xlabel('Rating')
plt.ylabel('Average Cost for two(In Indian Rupees)')
plt.title('Rating vs Avg. Cost for Two')
sns.pointplot(x='aggregate_rating',y='average_cost_for_two',data=new_df,ci=False)
plt.show()
plt.xlabel('Establishment')
plt.ylabel('Photo Count')
plt.title('Rating vs Avg. Cost for Two')
sns.boxplot(x='establishment',y='photo_count',data=new_df)
plt.xticks(rotation=90)
plt.show()
west_df=region_df[region_df['region']=='west']
west_feature=highlight_func(west_df)
west_feature=pd.DataFrame(west_feature)
west_feature.columns=['Highlight']
l=list(west_feature['Highlight'].unique())
print("There are ",len(l)," unique facilities available")
