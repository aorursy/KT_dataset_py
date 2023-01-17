
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import is_numeric_dtype
from scipy import stats
# Plotting Tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style("darkgrid")
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
#Subplots
from plotly.subplots import make_subplots
#search for missing data
import missingno as msno
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Bring test data into the environment
md= pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

#fuction to show more rows and columns
def show_all(df):
    #This fuction lets us view the full dataframe
    with pd.option_context('display.max_rows', 400, 'display.max_columns', 100):
        display(df)
show_all(md)
md.info()
md.describe().T
# Plot missing values of each column in the given dataset 
def plot_missing(df):
    # Find columns having missing values and count
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    # Plot missing values by count 
    missing.plot.bar(figsize=(16,5))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    msno.matrix(df=df, figsize=(16,5), color=(0,0.2,1))
plot_missing(md)
    
msno.heatmap(md,figsize=(16,8));
msno.dendrogram(md,figsize=(16,8));
ca = md[md['CouncilArea'].isnull()]
cb = md[md['CouncilArea'].notnull()]

for i in list(ca.index):
    if (ca['Postcode'][i] in list(cb['Postcode'])):
        x = cb[cb['Postcode']== ca['Postcode'][i]].index[0]
        ca['CouncilArea'][i] = cb.iloc[x]['CouncilArea']
md1=pd.merge(ca,cb, how ='outer') 
md1=md1[md1["CouncilArea"].notnull()]#3rows are deleted
md1=md1[md1["CouncilArea"] != 'Unavailable']#1 row is deleted
md1=md1[md1["CouncilArea"] != 'Moorabool']#1 row is deleted
a=md1.groupby(['CouncilArea'])['BuildingArea'].median()
ba= md1[md1['BuildingArea'].isnull()]
bb= md1[md1['BuildingArea'].notnull()]
for i in list(ba.index):
    j= ba['CouncilArea'][i]
    ba['BuildingArea'][i] = a[j]
md1=pd.merge(ba,bb, how ='outer') 
year=md1.groupby(['CouncilArea'])['YearBuilt'].median()
yeara= md1[md1['YearBuilt'].isnull()]
yearb= md1[md1['YearBuilt'].notnull()]
for i in list(yeara.index):
    j= yeara['CouncilArea'][i]
    yeara['YearBuilt'][i] = year[j]
md1=pd.merge(yeara,yearb, how ='outer')
car=md1.groupby(['CouncilArea'])['Car'].median()
cara= md1[md1['Car'].isnull()]
carb= md1[md1['Car'].notnull()]
for i in list(cara.index):
    j= cara['CouncilArea'][i]
    cara['Car'][i] = car[j]
md=pd.merge(cara,carb, how ='outer') 
import missingno as msno
msno.matrix(df=md, figsize=(16,5), color=(0,0.2,1));
fig = make_subplots(1,2)

fig.add_trace(go.Histogram(x=md1['Price']),1,1)
fig.add_trace(go.Box(y=md['Price'],boxpoints='all',line_color='purple'),1,2)

fig.update_layout(height=500, showlegend=False,title_text="SalePrice Distribution and Box Plot")
md['Price'].skew()
for name in list(md.columns):
    if is_numeric_dtype(md[name]):
        y = md[name]
        removed_outliers = y.between(y.quantile(.001), y.quantile(.999))
        #removed_outliers.value_counts()
        index_names = md[~removed_outliers].index # invert removed outliers
        md.drop(index_names, inplace=True)

show_all(md)
md['Date'] = pd.to_datetime(md['Date'])
md['Yr_sold'] = md['Date'].dt.year
md['Mth_sold'] = md['Date'].dt.month
date= md.groupby(['Yr_sold','Mth_sold'],as_index=False)['Price'].median()
yr_2016 = date[date['Yr_sold']==2016]
yr_2017 = date[date['Yr_sold']==2017]
fig = go.Figure()
fig.add_trace(go.Scatter(x=yr_2016['Mth_sold'], y=yr_2016['Price'],
                    mode='lines+markers',
                    name='House price in 2016'))
fig.add_trace(go.Scatter(x=yr_2017['Mth_sold'], y=yr_2017['Price'],
                    mode='lines+markers',
                    name='House price in 2017'))
fig.show()
#here we are checking the correlation between saleprice of house and other variables
corr_mat = md[['Price','Rooms','Distance','Bedroom2','Bathroom','Car','Landsize','Propertycount']].corr()
f, ax = plt.subplots(figsize=(30, 15))
sns.heatmap(corr_mat, vmax=1 , square=True,annot=True,linewidths=.5);


yrblt= md.groupby(['YearBuilt'],as_index=False)['Price'].median()
fig = go.Figure()
fig.add_trace(go.Scatter(x=yrblt['YearBuilt'], y=yrblt['Price'],
                    mode='lines+markers'))


fig.show()

#we can observe the level of influence of each variable in the following graphs.
cat1=['Rooms','Bedroom2','Bathroom','Car']

plt.figure(figsize=(25,15))
plt.subplots_adjust(hspace=0.5)

i = 1
for j in cat1:
    plt.subplot(1,4,i)
    sns.boxplot(x=md[j],y=md['Price'])
    plt.xlabel(j)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('SalePrice', fontsize=18)
    plt.xlabel(j, fontsize=18)
    i+=1
fig = px.histogram(md, x=md.Price, y=md.Landsize, color=md.Type,marginal="box", hover_data=md.columns)
fig.show()
fig=plt.figure(figsize=(15,8))

fig = sns.scatterplot(x='Propertycount', y='Price', data=md);
cat2=['Distance','Landsize','Propertycount']
sns.lmplot(x='Distance', y='Price', data=md,scatter=False,aspect=4,height=15);
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.ylabel('SalePrice', fontsize=50)
plt.xlabel('Distance', fontsize=50);
sns.lmplot(x="Landsize", y="Price", data=md,aspect=4);
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('SalePrice', fontsize=22)
plt.xlabel("Landsize", fontsize=22);

cat3=['Type','SellerG','CouncilArea','Regionname']
plt.figure(figsize=(26,40))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,1,1)
sns.barplot(data=md,x='Regionname',y='Price',hue="Type");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('Regionname', fontsize=18)
plt.xticks(rotation=90)
plt.xticks(rotation=90);
plt.subplot(2,1,2)
sns.barplot(data=md,x='CouncilArea',y='Price',hue="Type");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('CouncilArea', fontsize=18)
plt.xticks(rotation=90);

plt.figure(figsize=(26,40))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,1,1)
sns.swarmplot(data=md,x='Regionname',y='Price',hue="Method");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('Regionname', fontsize=18)
plt.xticks(rotation=90);
plt.subplot(2,1,2)
sns.swarmplot(data=md,x='CouncilArea',y='Price',hue="Method");
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(prop={'size': 15})
plt.ylabel('SalePrice', fontsize=18)
plt.xlabel('CouncilArea', fontsize=18)
plt.xticks(rotation=90);

a=md.groupby(['Suburb','Lattitude','Longtitude'],as_index=False)['Price'].median()
fig = px.scatter_mapbox(a,
                        lat="Lattitude",
                        lon="Longtitude",
                        color='Price',
                        mapbox_style='open-street-map',
                        hover_name='Suburb',
                        size='Price',
                        center={'lat': -37.8136, 'lon': 144.9631},
                        zoom=13,
                        hover_data=['Suburb','Price'],
                        title= 'SalesPrice In Each Suburb')
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(template='plotly_dark',margin=dict(l=20,r=20,t=40,b=20))
fig.show()