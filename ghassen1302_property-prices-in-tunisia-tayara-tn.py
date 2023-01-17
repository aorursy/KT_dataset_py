import pandas as pd

import numpy as np

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import folium

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('../input/property-prices-in-tunisia/Property Prices in Tunisia.csv')
report_v1 = pandas_profiling.ProfileReport(data)
report_v1.to_notebook_iframe()
plt.figure(figsize=(15,10))

sns.distplot(data.log_price,kde=False)
plt.figure(figsize=(15,10))

sns.boxplot(x='type',y='log_price', data=data)
plt.figure(figsize=(15,10))

sns.countplot(y='city', data=data, order=data.city.value_counts().index)
plt.figure(figsize=(15,10))

sns.countplot(y='category', data=data, order=data.category.value_counts().index)
sns.factorplot(data=data,x='log_price', y='city', hue='type', col='category',kind='bar', col_wrap=3)
g = sns.PairGrid(data, vars=['bathroom_count', 'log_price', 'size', 'room_count'],

                 hue='type', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend()
def hexbin(x, y, color, **kwargs):

    cmap = sns.light_palette(color, as_cmap=True)

    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)



with sns.axes_style("dark"):

    g1 = sns.FacetGrid(data, hue="type", col="type", height=4)

    g2 = sns.FacetGrid(data, hue="type", col="type", height=4)

    g3 = sns.FacetGrid(data, hue="type", col="type", height=4)



g1.map(hexbin, "room_count", "log_price", extent=[0, 50, 0, 10]);

g2.map(hexbin, "bathroom_count", "log_price", extent=[0, 6, 0, 10]);

g3.map(hexbin, "size", "log_price", extent=[0, 550, 0, 10]);
data2 = pd.DataFrame()

data2['city'] = data.city.unique()

data2 = data2.sort_values('city')

data2["lat"] = ""

data2["long"] = ""

for i in range(0, len(data2)):

    if(data2.iloc[i]["city"]=="Tunis"):

        data2["lat"].iloc[i] = 36.806112

        data2["long"].iloc[i] = 10.171078

        

    elif(data2.iloc[i]["city"]=="Ariana"):

        data2["lat"].iloc[i] = 36.860117

        data2["long"].iloc[i] = 10.193371

        

    elif(data2.iloc[i]["city"]=="Ben arous"):

        data2["lat"].iloc[i] = 36.753056

        data2["long"].iloc[i] = 10.218889

        

    elif(data2.iloc[i]["city"]=="La manouba"):

        data2["lat"].iloc[i] = 36.808029

        data2["long"].iloc[i] = 10.097205

        

    elif(data2.iloc[i]["city"]=="Nabeul"):

        data2["lat"].iloc[i] = 36.456058

        data2["long"].iloc[i] = 10.73763

    

    elif(data2.iloc[i]["city"]=="Zaghouan"):

        data2["lat"].iloc[i] = 36.402907

        data2["long"].iloc[i] = 10.142925

        

    elif(data2.iloc[i]["city"]=="Bizerte"):

        data2["lat"].iloc[i] = 37.274423

        data2["long"].iloc[i] = 9.87391

        

    elif(data2.iloc[i]["city"]=="Béja"):

        data2["lat"].iloc[i] = 36.725638

        data2["long"].iloc[i] = 9.181692

    

    elif(data2.iloc[i]["city"]=="Jendouba"):

        data2["lat"].iloc[i] = 36.501136

        data2["long"].iloc[i] = 8.780239

    

    elif(data2.iloc[i]["city"]=="Le kef"):

        data2["lat"].iloc[i] = 36.174239

        data2["long"].iloc[i] = 8.704863

    

    elif(data2.iloc[i]["city"]=="Siliana"):

        data2["lat"].iloc[i] = 36.084966

        data2["long"].iloc[i] = 9.370818

    

    elif(data2.iloc[i]["city"]=="Sousse"):

        data2["lat"].iloc[i] = 35.825388

        data2["long"].iloc[i] = 10.636991

    

    elif(data2.iloc[i]["city"]=="Monastir"):

        data2["lat"].iloc[i] = 35.783333

        data2["long"].iloc[i] = 10.833333

    

    elif(data2.iloc[i]["city"]=="Mahdia"):

        data2["lat"].iloc[i] = 35.504722

        data2["long"].iloc[i] = 11.062222

    

    elif(data2.iloc[i]["city"]=="Sfax"):

        data2["lat"].iloc[i] = 34.740556

        data2["long"].iloc[i] = 10.760278

    

    elif(data2.iloc[i]["city"]=="Kairouan"):

        data2["lat"].iloc[i] = 35.678102

        data2["long"].iloc[i] = 10.096333

    

    elif(data2.iloc[i]["city"]=="Kasserine"):

        data2["lat"].iloc[i] = 35.167578

        data2["long"].iloc[i] = 8.836506

    

    elif(data2.iloc[i]["city"]=="Sidi bouzid"):

        data2["lat"].iloc[i] = 35.038234

        data2["long"].iloc[i] = 9.484935

    

    elif(data2.iloc[i]["city"]=="Gabès"):

        data2["lat"].iloc[i] = 33.881457

        data2["long"].iloc[i] = 10.098196

    

    elif(data2.iloc[i]["city"]=="Médenine"):

        data2["lat"].iloc[i] = 33.354947

        data2["long"].iloc[i] = 10.505478

    

    elif(data2.iloc[i]["city"]=="Tataouine"):

        data2["lat"].iloc[i] = 32.929674

        data2["long"].iloc[i] = 10.451767

    

    elif(data2.iloc[i]["city"]=="Gafsa"):

        data2["lat"].iloc[i] = 34.425

        data2["long"].iloc[i] = 8.784167

    

    elif(data2.iloc[i]["city"]=="Tozeur"):

        data2["lat"].iloc[i] = 33.919683

        data2["long"].iloc[i] = 8.13352

    

    else:

        data2["lat"].iloc[i] = 33.704387

        data2["long"].iloc[i] = 8.969034



data2.reset_index(drop=True, inplace=True)

data2['sum_room_count'] = data.groupby(['city'])['room_count'].sum().reset_index(name='sum_room_count')['sum_room_count']

data2['sum_bathroom_count'] = data.groupby(['city'])['bathroom_count'].sum().reset_index(name='sum_bathroom_count')['sum_bathroom_count']

data2['sum_size'] = data.groupby(['city'])['size'].sum().reset_index(name='sum_size')['sum_size']

data2['sum_log_price'] = data.groupby(['city'])['log_price'].sum().reset_index(name='sum_log_price')['sum_log_price']



data2['sum_À Vendre'] = data[data["type"]=='À Vendre'].groupby(['city'])['type'].count().reset_index(name='sum_À Vendre')['sum_À Vendre']



x=data[data["type"]=='À Louer'].groupby(['city'])['type'].count().reset_index(name='sum_À Louer')

for i in range (len(data.city.unique())):

    if(data.city.unique()[i] not in np.array(data[data["type"]=='À Louer'].groupby(['city'])['type'].count().reset_index(name='sum_À Louer')['city'])):

        x=x.append(pd.DataFrame([[data.city.unique()[i], 0]], columns=['city','sum_À Louer']),ignore_index=True)

x = x.sort_values('city')

x.reset_index(drop=True, inplace=True)

data2['sum_À Louer'] = x['sum_À Louer']

    

for c in (['Appartements', 'Bureaux et Plateaux', 'Colocations', 'Locations de vacances', 'Magasins, Commerces et Locaux industriels', 'Maisons et Villas', 'Terrains et Fermes' ]):

    x=data[data["category"]==c].groupby(['city'])['category'].count().reset_index(name='sum_'+c)

    for i in range (len(data.city.unique())):

        if(data.city.unique()[i] not in np.array(data[data["category"]==c].groupby(['city'])['category'].count().reset_index(name='sum_'+c)['city'])):

            x=x.append(pd.DataFrame([[data.city.unique()[i], 0]], columns=['city','sum_'+c]),ignore_index=True)

    x = x.sort_values('city')

    x.reset_index(drop=True, inplace=True)

    data2['sum_'+c] = x['sum_'+c]



data2 = data2.fillna(0)

num = data2._get_numeric_data()

num[num < 0] = 0
data2
m = folium.Map(location=[33.8869, 10], tiles='cartodbpositron', zoom_start=6)



for i in range(0, len(data2)):

    

    folium.Circle(

        location=[data2.iloc[i]['lat'], data2.iloc[i]['long']],

        color='crimson',

        tooltip =   '<li><bold>City : '+str(data2.iloc[i]['city'])+

                    '<li><bold>sum(log(Prices)) : '+str(data2.iloc[i]['sum_log_price'])+

                    '<li><bold>sum(For sale) : '+str(data2.iloc[i]['sum_À Vendre'])+

                    '<li><bold>sum(For rent) : '+str(data2.iloc[i]['sum_À Louer'])+

                    '<li><bold>sum(Apartments) : '+str(data2.iloc[i]['sum_Appartements'])+

                    '<li><bold>sum(Offices) : '+str(data2.iloc[i]['sum_Bureaux et Plateaux'])+

                    '<li><bold>sum(Shared flat) : '+str(data2.iloc[i]['sum_Colocations'])+

                    '<li><bold>sum(Holiday rents) : '+str(data2.iloc[i]['sum_Locations de vacances'])+

                    '<li><bold>sum(Shops, Stores and Industrial) : '+str(data2.iloc[i]['sum_Magasins, Commerces et Locaux industriels'])+

                    '<li><bold>sum(Houses and Villas) : '+str(data2.iloc[i]['sum_Maisons et Villas'])+

                    '<li><bold>sum(Land and Farms) : '+str(data2.iloc[i]['sum_Terrains et Fermes'])+

                    '<li><bold>sum(Rooms) : '+str(data2.iloc[i]['sum_room_count'])+

                    '<li><bold>sum(Bathrooms) : '+str(data2.iloc[i]['sum_bathroom_count'])+

                    '<li><bold>sum(Size) : '+str(data2.iloc[i]['sum_size']),

        

        radius=int(data2.iloc[i]['sum_log_price'])**1).add_to(m)

    

m