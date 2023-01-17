import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

from folium.plugins import FastMarkerCluster

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



pd.set_option('display.max_rows', 500)
#Import the dataset and keep relevant columns only

data=pd.read_csv("/kaggle/input/fast-food-restaurants/FastFoodRestaurants.csv")

del data['websites']

data.drop(columns=['keys'],inplace=True)

data.head(5)

lats = data['latitude'].tolist()

lons = data['longitude'].tolist()

locations = list(zip(lats, lons))



map1 = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

FastMarkerCluster(data=locations).add_to(map1)

map1
def market(x):

    

    if x in ('WA','OR','CA','NV','AZ','NM','CO','WY','MT','ID','AK','HI'):

        return "West"

    elif x in ('ND','MN','IA','SD','KS','NE','MO','WI','MI','IL','IN','OH'):

        return "Mid West"

    elif x in ('NY','NJ','PA','CT','RI','MA','ME','NH','VT'):

        return "North East"

    else:

        return "South"



data['region']=data['province'].apply(market)

data.head(5)
d1=data.groupby(['region','province']).agg({'name':'count'})

d1.columns=['Perc_Region']

d1=d1.groupby(level=0).apply(lambda x: 100*x/x.sum())

d1.reset_index(inplace=True)

d1['Perc_Region']=d1['Perc_Region'].round(1)





d1.sort_values(['region','Perc_Region'],ascending=[False,False],inplace=True)

d1.set_index(['region','province'])

d1.reset_index(inplace=True)



d11=d1[d1.region.isin(['West'])].sort_values(['Perc_Region'],ascending=True)

d12=d1[d1.region.isin(['Mid West'])].sort_values(['Perc_Region'],ascending=True)

d13=d1[d1.region.isin(['North East'])].sort_values(['Perc_Region'],ascending=True)

d14=d1[d1.region.isin(['South'])].sort_values(['Perc_Region'],ascending=True)



# d1.head(5)



fig=px.bar(d1,x=d1['province'],y=['Perc_Region'],color='region',title="Fast Food Restaurant Presence (%): Percentages are at Regional Level and not Overall")

fig.update_layout(xaxis_title="State",yaxis_title="Percentage")

fig.show()
state_counts=data.groupby(['province']).agg({'name':'count'})

state_counts.columns=['Count of Brands']

state_counts.sort_values('Count of Brands',ascending=False,inplace=True)

state_counts.reset_index(inplace=True)



state_counts['Perc']=100*state_counts['Count of Brands']/state_counts['Count of Brands'].sum()

# state_counts.head(10)



fig=px.bar(state_counts,x="province",y="Perc")

fig.update_traces(marker_color='Turquoise')

fig.update_layout(xaxis_title="State",yaxis_title="Percentage",title="Fast Food Restaurant Presence(%)")

fig.show()
region_counts=data.groupby(['region']).agg({'name':'count'})

region_counts.columns=['Count of Brands']



region_counts.sort_values('Count of Brands', ascending=False,inplace=True)

region_counts['perc']=100*region_counts['Count of Brands']/region_counts['Count of Brands'].sum()

region_counts.reset_index(inplace=True)

# print(region_counts)



fig=px.pie(region_counts,values='perc',names='region',title='Brand Presence by Region',width=600,height=500,

           color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()


data['name']=data.name.str.replace('$','')

data['name']=data.name.str.replace("'",'')

data['name']=data.name.str.lower()    

data['name']=data.name.map(lambda x: re.sub(r'\W+','',x))





Brand_1=data.groupby(['name']).agg({'name':'count'})

Brand_1.columns=['number']

Brand_1=Brand_1.sort_values('number',ascending=False)



# Brand_1.head(10)

Brand_1.nlargest(15,'number').index
Brands=['mcdonalds', 'burgerking', 'tacobell', 'wendys', 'arbys', 'kfc','subway']

data1=data[data.name.isin(Brands)]



# print(data1.shape)

data1.head(10)



d1=data1.groupby(['region','name']).agg({'name':'count'})

d1.columns=['Stores']

d1.reset_index(inplace=True)

d1.sort_values(['region','Stores'],ascending=['True','False'],inplace=True)

d1.reset_index(inplace=True)

d1.drop('index',inplace=True,axis=1)



import plotly.graph_objects as go

import plotly.express as px



fig = px.scatter(d1, x="Stores", y="name", color="region",

                 title="Brand Presence by Region-Store Counts",

                 labels={"Stores":"Store Count"} 

                )



fig.update_layout(

    margin=dict(l=20, r=20, t=25, b=20)

#     paper_bgcolor="LightSteelBlue",

)

fig.show()
d2=d1.copy()

d2.set_index(['region','name'],inplace=True)

d2['Percentage']=d2['Stores'].groupby(level=0).apply(lambda x: 100*x/x.sum())

# del d2['perc']

d2.reset_index(inplace=True)



# d2.reset_index(inplace=True)

fig2=px.histogram(d2,x="region",y="Percentage",color="name",width=800,height=500,

                 title="Brand Presence by Region")

# fig2.update_layout(margin=dict(l=))

fig2.show()
d3=data1.groupby(['province','name']).agg({'name':'count'})

d3.columns=['Stores']

d3['Percentage']=d3['Stores'].groupby(level=0).apply(lambda x:100*x/x.sum())

d3.reset_index(inplace=True)

# d3



fig4=px.histogram(d3,x="province",y="Percentage",color="name",

                 title="Brand Penetration by State")

fig4.show()
dh=pd.read_csv('/kaggle/input/la-restaurant-market-health-data/restaurant-and-market-health-violations.csv')

# dh.columns

cols=['facility_name', 'violation_code',

       'violation_description', 'violation_status', 'points', 'grade',

       'facility_address', 'facility_city', 'facility_id', 'facility_state',

       'facility_zip','score', 'service_code',

       'service_description', 'row_id']

dh=dh[[x for x in cols]]

dh.drop_duplicates(keep=False,inplace=True)

# print(dh.shape)

dh.shape
# View top 5 records 

dh.head(5)
# Cleaning up facility names by removing special characters, white spaces and converting to lower

dh['facility_name']=dh.facility_name.apply(lambda x: re.sub(r'\W+','',x))

dh['facility_name']=dh.facility_name.str.lower()

dh[['facility_name','violation_code','violation_description']].head(5)
# Obtain Top Health Violations 

dh1=dh['violation_description'].value_counts().sort_values(ascending=True).to_frame()

dh1.reset_index(inplace=True)

dh1.columns=['Violation Description','Number of Violations']

# dh1.tail(10)



fig11=px.bar(dh1[(dh1.shape[0]-10):],x='Number of Violations',y='Violation Description',orientation='h',height=500,width=1200)

fig11.update_layout(xaxis_title="Number of Violations",yaxis_title="Violation Code",title="Common Health Violations in LA")

fig11.update_traces(marker_color='darkcyan')

fig11.show()
dh2=dh['facility_name'].value_counts().sort_values(ascending=False).to_frame()

dh2.reset_index(inplace=True)

dh2.columns=['facility_name','Nbr of Violations']

# print(f'There are a total of {len(dh2)} Unique Facilities in dataset')



# selecting focus brands only

brands=['burgerki','donalds','subway','tacobell','kfc','arbys','wendys']

# dh2[dh2['facility_name'].str.contains(r'burgerking|donalds')]

dh2=dh2[dh2['facility_name'].str.contains('|'.join(map(re.escape, brands)))]

# dh2.head(5)



# we observe that brand names are combined with numeric data henceforth further refining

dh2.loc[dh2['facility_name'].str.contains(r'burgerking'),'Brand']="BurgerKing"

dh2.loc[dh2['facility_name'].str.contains(r'wendys'),'Brand']="Wendys"

dh2.loc[dh2['facility_name'].str.contains(r'mcdonalds'),'Brand']="McDonalds"

dh2.loc[dh2['facility_name'].str.contains(r'subway'),'Brand']="Subway"

dh2.loc[dh2['facility_name'].str.contains(r'kfc'),'Brand']="KFC"

dh2.loc[dh2['facility_name'].str.contains(r'tacobell'),'Brand']="TacoBell"

dh2.loc[dh2['facility_name'].str.contains(r'arbys'),'Brand']="Arbys"



# group at brand level



dh3=dh2[['Brand','Nbr of Violations','facility_name']].groupby(['Brand']).agg({'Nbr of Violations':'sum',

                                'facility_name':'count'})

dh3.reset_index(inplace=True)

dh3.columns=['Brands','Nbr_of_Violations','Stores']

# Normalizing Violations across Brands

dh3['Violations_per_store']=dh3['Nbr_of_Violations']/dh3['Stores']

dh3.sort_values('Violations_per_store',ascending=False,inplace=True)



fig5=make_subplots(rows=1,cols=3,subplot_titles=("Total Violations","Store Counts","Violations Per Store"))



trace_1=go.Bar(x=dh3.Brands,y=dh3.Nbr_of_Violations,name='Total Violations')

trace_2=go.Bar(x=dh3.Brands,y=dh3.Stores,name='Store Counts')

trace_3=go.Bar(x=dh3.Brands,y=dh3.Violations_per_store,name='Violations Per Store')



fig5.append_trace(trace_1,1,1)

fig5.append_trace(trace_2,1,2)

fig5.append_trace(trace_3,1,3)



fig5.update_layout(title='Health Inspection Results by Brands- Los Angeles',showlegend=False,height=450)



fig5.show()
df=pd.read_csv('/kaggle/input/chicago-food-inspections/food-inspections.csv')

df.drop_duplicates(keep=False,inplace=True)

df.columns
df=df[['DBA Name', 'AKA Name', 'Facility Type','Address',

       'Risk', 'City', 'State', 'Zip', 'Inspection Date',

       'Inspection Type', 'Results','Violations']]

df.head(5)
print(f'the violations description is very specific. There are {len(df.Violations.unique())} unique violations')
# Since Violation descriptions are unique we are selecting only first 75 characters from the string



df['Violations2']=df['Violations'].str[:75]

df.drop('Violations',1,inplace=True)



df['DBA Name']=df['DBA Name']+df['Address']

df=df[df['Results']=='Fail']

df1=df[['Violations2']].groupby(['Violations2']).agg({'Violations2':'count'})

df1.columns=['Counts']

df1.reset_index(inplace=True)

df1.columns=['Violations','Counts']

df1.sort_values('Counts',ascending=True,inplace=True)

# df1.head(10)



fig6=px.bar(df1[df1.shape[0]-10:],x='Counts',y='Violations',orientation='h',height=500,width=1500)

fig6.update_layout(xaxis_title="Number of Violations",yaxis_title="Violation Code",title="Common Health Violations in Chicago")

fig6.update_traces(marker_color='darkcyan')

fig6.show()
df2=df.copy()

df2['DBA Name']=df2['DBA Name'].str.lower()

df2['DBA Name']=df2['DBA Name'].apply(lambda x: re.sub(r'\W+','',x))



df2=df2['DBA Name'].value_counts().sort_values(ascending=False).to_frame()

df2.reset_index(inplace=True)

df2.columns=['DBA Name','Nbr of Violations']

# print(f'There are a total of {len(dh2)} Unique Facilities in dataset')





brands=['burgerki','donalds','subway','tacobell','kfc','arbys','wendys']

df2=df2[df2['DBA Name'].str.contains('|'.join(brands))]



df2.loc[df2['DBA Name'].str.contains(r'burgerk'),'Brand']='BurgerKing'

df2.loc[df2['DBA Name'].str.contains(r'donalds'),'Brand']='McDonalds'

df2.loc[df2['DBA Name'].str.contains(r'arby'),'Brand']='Arbs'

df2.loc[df2['DBA Name'].str.contains(r'subway'),'Brand']='Subway'

df2.loc[df2['DBA Name'].str.contains(r'tacobe'),'Brand']='TacoBell'

df2.loc[df2['DBA Name'].str.contains(r'wendy'),'Brand']='Wendys'

df2.loc[df2['DBA Name'].str.contains(r'kfc'),'Brand']='KFC'

# df2.head(10)



df3=df2[['Brand','Nbr of Violations','DBA Name']].groupby(['Brand']).agg({'Nbr of Violations':'sum',

                                'DBA Name':'count'})

df3.reset_index(inplace=True)

df3.columns=['Brands','Nbr_of_Violations','Stores']



# Normalizing Violations across Brands

df3['Violations_per_store']=df3['Nbr_of_Violations']/df3['Stores']

df3.sort_values('Violations_per_store',ascending=False,inplace=True)

# df3.head(5)



fig7=make_subplots(rows=1,cols=3,subplot_titles=("Total Violations","Store Counts","Violations Per Store"))



trace_1=go.Bar(x=df3.Brands,y=df3.Nbr_of_Violations,name='Total Violations')

trace_2=go.Bar(x=df3.Brands,y=df3.Stores,name='Store Counts')

trace_3=go.Bar(x=df3.Brands,y=df3.Violations_per_store,name='Violations Per Store')



fig7.append_trace(trace_1,1,1)

fig7.append_trace(trace_2,1,2)

fig7.append_trace(trace_3,1,3)



fig7.update_layout(title='Health Inspection Results by Brands- Chicago',showlegend=False,height=450)



fig7.show()
# Import json data into dataframe format and consider focus brands

b1= pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_business.json", lines=True, orient='columns', chunksize=1000000)

for x in b1:

    b2 = x

    break



b3=b2.copy()

b3['name']=b3.name.str.lower()

b3['name']=b3['name'].apply(lambda x: re.sub('\W+','',x))

brands=['burgerking','kfc','wendys','arbys','mcdonalds','subway','tacobell']

b4=b3[b3['name'].str.match("|".join(brands))]



#Define new column as Brand

b4.loc[b4['name'].str.contains(r'wendys'),'Brand']='Wendys'

b4.loc[b4['name'].str.contains(r'burgerk'),'Brand']='Burgerking'

b4.loc[b4['name'].str.contains(r'kfc'),'Brand']='KFC'

b4.loc[b4['name'].str.contains(r'tacobe'),'Brand']='TacoBell'

b4.loc[b4['name'].str.contains(r'subway'),'Brand']='Subway'

b4.loc[b4['name'].str.contains(r'arbys'),'Brand']='Arbys'

b4.loc[b4['name'].str.contains(r'donalds'),'Brand']='McDonalds'



b4['Market']=b4['state'].apply(market)

b4.head(5)
fig8=px.histogram(b4,x='stars',color='Brand',facet_col='Brand',height=450)

fig8.update_layout(showlegend=False,title='Yelp Ratings')

fig8.show()
b5=b4.groupby(['Brand','stars']).agg({'name':'count'})

b5=b5.groupby(level=0).apply(lambda x: 100*x/x.sum())

b5.reset_index(inplace=True)

b5.columns=['Brand','stars','Perc']

b5['Perc']=b5['Perc'].round(1)



b5['stars']=b5['stars'].astype(str)

fig9=px.bar(b5,x='Brand',y='Perc',color='stars',labels={'stars':'Ratings','Perc':'% of Ratings'},

            color_continuous_scale=px.colors.sequential.Viridis,height=450,width=1100)

fig9.update_layout(title="Ratings Distribution")

fig9.show()
# !jupyter nbconvert --execute --to html FastFoodRestaurantsMarketAnalysis.ipynb

# !jupyter nbconvert  FastFoodRestaurantsMarketAnalysis.ipynb --to html



!jupyter nbconvert --to html FastFoodRestaurantsMarketAnalysis.ipynb