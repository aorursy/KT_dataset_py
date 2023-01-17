import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



#Plotly

import plotly.express as px

import plotly.graph_objs as go



#Some styling

sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")



import plotly.io as pio

pio.templates.default = 'presentation'



#Subplots

from plotly.subplots import make_subplots





#Showing full path of datasets

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/melbourne-housing-snapshot/melb_data.csv")
df.head()
df.shape
df.columns
#Firstly I will change some column names to make them more meaningful

# It is not required but it helps me in better analysis.



df.rename(columns={"Type":"Property_Type",

                  "Method":"Method_Sold",

                  "Distance":"Distance_CBD",

                  "Car":"Carspots",

                  "Date":"Date_Sold"},inplace=True)
#After changing few column names

df.columns



#Now we have some meaningful names
#Let's look at the info again



df.info()
df.describe().T
#Let's check the skewness of our features



df.skew()
df.isna().sum().sum()
df.isna().sum()
df['Carspots'].isna().sum()
df["Carspots"].median()
df['Carspots'].value_counts()
df['Carspots'] = df['Carspots'].fillna(0.0)
df['BuildingArea'].isna().sum()
df.drop(columns=['BuildingArea'],axis=1,inplace=True)
df['YearBuilt'].isna().sum()
df['YearBuilt'] = df['YearBuilt'].fillna(1190)
df['YearBuilt'] = df['YearBuilt'].astype(int)
pd.to_datetime(df['YearBuilt'], format='%Y',errors = 'coerce').dt.to_period('Y')

df['CouncilArea'].isna().sum()
df['CouncilArea'].value_counts()
df['CouncilArea'] = df['CouncilArea'].fillna('Unavailable')
df['Landsize'].mean()
df['Landsize'].median()
#First step is to find outliers in our data

# Then decide whether to remove the outliers or to cap them

# We exclude here categorical and discrete features.

# And datetime related features ,Lattitude and Logtitude

# With this , we are left with only one feature i,e Landsize



col ='Landsize'





IQR = df[col].quantile(0.75) - df[col].quantile(0.25)

Lower_Bound = df[col].quantile(0.25) - (IQR*1.5)

Upper_Bound = df[col].quantile(0.25) + (IQR*1.5)



print("The outliers in {} feature are values << {} and >> {}\n".format(col,Lower_Bound,Upper_Bound))

minimum=df[col].min()

maximum=df[col].max()

print("The minimum value in {} is {} and maximum value is {}".format(col,minimum,maximum))



print("\nMaximum value is greater than the Upper_Bound limit")

print("Thus , outliers are values greater than Upper_Bound")



number_of_out = len(df[df['Landsize']>Upper_Bound])



print('\nThere are {} outliers in Landsize feature'.format(number_of_out))



        

    
# We'll look at the box plot of Landsize before and after capping the outliers to show the difference.



fig = px.box(df,y='Landsize',width=800,title='Before capping the outliers')

fig.show()
df['Landsize'] = np.where(df[col]>Upper_Bound,Upper_Bound,df[col])
fig = px.box(df,y='Landsize',width=800,title='After capping the outliers')

fig.show()
fig=plt.figure(figsize=(15,10))



plt.subplot(2,1,1)

fig1 = sns.distplot(df['Price'],color='red')



plt.subplot(2,1,2)

fig2 = sns.boxplot(data=df,x='Price',color='aqua')
#Extracting the categorical features from the dataset

skip_features=['Date_Sold','YearBuilt']  #Analyze both features later

cat = [col for col in df.columns if df[col].dtype=="O" and col not in skip_features]
#Display categorical features

print(cat)
df['Suburb'].describe()
df['Suburb'].value_counts()
df.groupby(['Suburb','Regionname'],as_index=False)['Lattitude','Longtitude','Price'].median()
df.columns
map_suburb = df.groupby(['Suburb','Regionname'],as_index=False)['Lattitude','Longtitude','Price'].median()

map_suburb



fig = px.scatter_mapbox(map_suburb,

                        lat="Lattitude",

                        lon="Longtitude",

                        color='Price',

                        mapbox_style='open-street-map',

                        hover_name='Suburb',

                        color_continuous_scale=px.colors.cyclical.IceFire,

                        size='Price',

                        center={"lat": -37.8136, "lon": 144.9631},

                        zoom=12,

                        hover_data=['Regionname','Suburb','Price'],

                       title='Average Price in different suburbs and regions')

fig.update_geos(fitbounds="locations", visible=True)

fig.update_geos(projection_type="orthographic")

fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))

fig.show()

df['Property_Type'].unique()
df['Property_Type'].value_counts()
sns.countplot(data=df,x='Property_Type')
#We groupby the data with Property Type and take median of Sale Price



temp = df.groupby(['Property_Type'],as_index=False)['Price'].median()

fig = px.bar(temp.sort_values(by='Price'),y='Property_Type',x='Price',orientation='h',color='Price',text='Price')

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',height=400,width=850)



fig.show()
townhouse = df[df['Property_Type']=='t']

townhouse = townhouse.reset_index(drop=True)

px.bar(townhouse.groupby(['Regionname'],as_index=False)['Price'].median().sort_values(by='Price',ascending=False),y='Price',x='Regionname',color='Price',

       color_continuous_scale='Rainbow',height=500,width=800)
df['Method_Sold'].unique()
df['Method_Sold'].value_counts()
fig1 = px.histogram(df,x="Price",color='Method_Sold',barmode='overlay')

fig1.show()



fig2 = px.box(df,x='Method_Sold',y='Price',color='Method_Sold')

fig2.show()
df['SellerG'].nunique()
sellers = df.groupby(['SellerG'],as_index=False)['Price'].median()

sellers = sellers.sort_values(by='Price',ascending=False).reset_index(drop=True)

sellers
sellers_pt = df.groupby(['SellerG','Property_Type'],as_index=False)['Price','Lattitude','Longtitude'].median()

sellers_pt
# Now we separate the sellers into three property types,

sellers_pt_h = sellers_pt[sellers_pt['Property_Type']=='h']

sellers_pt_u = sellers_pt[sellers_pt['Property_Type']=='u']

sellers_pt_t = sellers_pt[sellers_pt['Property_Type']=='t']
fig = px.scatter_mapbox(sellers_pt_h,

                        lat="Lattitude",

                        lon="Longtitude",

                        color='Price',

                        mapbox_style='open-street-map',

                        hover_name='SellerG',

                        color_continuous_scale=px.colors.cyclical.IceFire,

                        size='Price',

                        center={"lat": -37.8136, "lon": 144.9631},

                        zoom=12,

                        hover_data=['SellerG','Property_Type','Price'],

                       title='Average price of houses sold by sellers')

fig.update_geos(fitbounds="locations", visible=True)

fig.update_geos(projection_type="orthographic")

fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))

fig.show()
fig = px.scatter_mapbox(sellers_pt_u,

                        lat="Lattitude",

                        lon="Longtitude",

                        color='Price',

                        mapbox_style='open-street-map',

                        hover_name='SellerG',

                        color_continuous_scale=px.colors.cyclical.IceFire,

                        size='Price',

                        center={"lat": -37.8136, "lon": 144.9631},

                        zoom=12,

                        hover_data=['SellerG','Property_Type','Price'],

                       title='Average price of units sold by sellers')

fig.update_geos(fitbounds="locations", visible=True)

fig.update_geos(projection_type="orthographic")

fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))

fig.show()
fig = px.scatter_mapbox(sellers_pt_t,

                        lat="Lattitude",

                        lon="Longtitude",

                        color='Price',

                        mapbox_style='open-street-map',

                        hover_name='SellerG',

                        color_continuous_scale=px.colors.cyclical.IceFire,

                        size='Price',

                        center={"lat": -37.8136, "lon": 144.9631},

                        zoom=12,

                        hover_data=['SellerG','Property_Type','Price'],

                       title='Average price of towhouses sold by sellers')

fig.update_geos(fitbounds="locations", visible=True)

fig.update_geos(projection_type="orthographic")

fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))

fig.show()
top_5 = df.query('SellerG in ["Weast","VICProp","Darras","Lucas","Kelly"]')

top_5
df[df['Price']==9000000]['SellerG']
temp[-5:]
bottom_5 = df.query('SellerG in ["Wood","hockingstuart/Village","hockingstuart/Advantage","Rosin","PRDNationwide"]')

bottom_5
df[df['Price']==85000]['SellerG']
df['CouncilArea'].value_counts()
fig=plt.figure(figsize=(20,25))



table = df.groupby(['CouncilArea'],as_index=False)['Price'].median().sort_values(by='Price',ascending=False)



sns.boxplot(data=df,x='Price',y='CouncilArea',order=table['CouncilArea'].to_list());

plt.yticks(fontsize=18);

plt.xticks(fontsize=18);

plt.ylabel("Council Areas",fontsize=22);

plt.xlabel("Price",fontsize=22);

df['Regionname'].value_counts()
fig=px.histogram(df,x='Price',color='Regionname',barmode="overlay")

fig.show()



fig=plt.figure(figsize=(15,10))

fig=sns.boxplot(data=df,x='Price',y='Regionname',palette='Dark2')

plt.yticks(fontsize=18);

plt.xticks(fontsize=18);

plt.ylabel("",fontsize=22);

plt.xlabel("Price",fontsize=22);

plt.show()
fig = px.box(df,y='Price',x='Regionname',color='Property_Type',height=800,width=1500)



fig.update_xaxes(

    showgrid=True,

    tickson="boundaries",

    ticklen=10

)
num=[col for col in df.columns if df[col].dtype!="O"]
print(num)
discrete=[]

for col in df.columns:

    if df[col].dtype!="O" and len(df[col].unique()) < 15:

        discrete.append(col)
print(discrete)
fig = plt.figure(figsize=(15,5))

fig = sns.boxplot(data=df,x='Rooms',y='Price',palette='Purples')

plt.show()



fig = plt.figure(figsize=(15,5))

fig = sns.boxplot(data=df,x='Bedroom2',y='Price',palette='Reds')

plt.show()



fig = plt.figure(figsize=(15,5))

fig = sns.boxplot(data=df,x='Bathroom',y='Price',palette='Greens')

plt.show()



fig = plt.figure(figsize=(15,5))

fig = sns.boxplot(data=df,x='Carspots',y='Price',palette='Blues')

plt.show()
continuous = [col for col in df.columns if df[col].dtype!="O" and col not in discrete]



print(continuous)
continuous = ['Distance_CBD', 'Postcode', 'Landsize', 'Propertycount','Price']
corr = df[continuous].corr()

fig = plt.figure(figsize=(15,10))



sns.heatmap(corr,annot=True,linewidths=.5,cmap='coolwarm',vmin=-1,vmax=1,center=0);
#Separate Price as we have already been analyzed

#And also separate Postcode as it is not useful

continuous = ['Distance_CBD', 'Landsize', 'Propertycount']



plt.figure(figsize=(15,5))

plt.subplots_adjust(hspace=0.2)



i=1

colors = ['indianred','chocolate','yellowgreen','indigo']

j=0

for col in continuous:

    plt.subplot(1,3,i)

    a1 = sns.distplot(df[col],color=colors[j])

    i+=1

    j+=1
continuous = ['Distance_CBD', 'Landsize', 'Propertycount']





plt.figure(figsize=(15,5))

i=1

colors = ['indianred','chocolate','yellowgreen']

j=0

for col in continuous:

    plt.subplot(1,3,i)

    a1 = sns.scatterplot(data=df,x=col,y='Price',color=colors[j])

    i+=1

    j+=1
df['Date_Sold'] = pd.to_datetime(df['Date_Sold'])

df['Year_Sold'] = df['Date_Sold'].dt.year
year_sold_grouped = df.groupby(['Year_Sold'],as_index=False)['Price'].median()

year_sold_grouped
df['Month_Sold'] = df['Date_Sold'].dt.month

df['Month_Sold']
year_sold = df.groupby(['Year_Sold','Month_Sold'],as_index=False)['Price'].median()

year_sold
year_sold_2016 = year_sold[year_sold['Year_Sold']==2016]

year_sold_2017 = year_sold[year_sold['Year_Sold']==2017]
fig = go.Figure()

fig.add_trace(go.Scatter(x=year_sold_2016['Month_Sold'], y=year_sold_2016['Price'],

                    mode='lines+markers',

                    name='House sold in 2016'))



fig.add_trace(go.Scatter(x=year_sold_2017['Month_Sold'], y=year_sold_2017['Price'],

                    mode='lines+markers',

                    name='House sold in 2017'))

fig.show()
temp = df.groupby(['YearBuilt'],as_index=False)['Price'].median()



#We drop first two rows as one is the missing values and other is an outlier in our YearBuilt feature.

temp = temp.drop([0,1],axis=0).reset_index(drop=True)

temp
fig = go.Figure()

fig.add_trace(go.Scatter(x=temp['YearBuilt'], y=temp['Price'],

                    mode='lines+markers',

                    name='Average Price over the years'))



fig.show()