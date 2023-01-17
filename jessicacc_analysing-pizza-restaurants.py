# Import packages

import pandas as pd

import numpy as np

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline

import plotly.offline as py
pizzas = pd.read_csv('../input/8358_1.csv')
pizzas.head(2)
pizzas.shape
# Check null values

pizzas.isnull().sum()
# removes unnecessary variables for analysis

inuteis = ['menuPageURL', 'id']

pizzas = pizzas.drop(inuteis, axis=1)  



# Increase null values

pizzas['menus.currency'].fillna('USD',inplace=True)

pizzas['priceRangeCurrency'].fillna('USD',inplace=True)



# Increase values with average

pizzas['priceRangeMin'].fillna(round(pizzas['priceRangeMin'].mean(), 2),inplace=True)

pizzas['priceRangeMax'].fillna(round(pizzas['priceRangeMax'].mean(), 2),inplace=True)
# Pizza flavors

sabores = pizzas['menus.name'].value_counts() 



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

sabores[:10].plot.bar()

plt.title('10 most popular pizzas')

plt.ylabel('Quantity')      

plt.show();
print(sabores.head(30))
# Pizza Burger

pb_df = pizzas[pizzas['menus.name']=='Pizza Burger'] #Dataframe only for Pizza Burger



# Map Parameters

data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = pb_df['longitude'],

        lat = pb_df['latitude'],

        text = pb_df['city'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.6,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            )))]



# Layout

layout = dict(

        title = 'Pizza Burger Locations',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa'),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5))



py.init_notebook_mode(connected=True)

fig = dict( data=data, layout=layout )

py.iplot(fig, filename='pizzasBurger.html')
# White Pizza

wp_df = pizzas[pizzas['menus.name']=='White Pizza'] #Dataframe oly for White Pizza



# Map Parameters

data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = wp_df['longitude'],

        lat = wp_df['latitude'],

        text = wp_df['city'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.6,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            )))]



# Layout

layout = dict(

        title = 'Locations of White Pizzas',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa'),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5))



py.init_notebook_mode(connected=True)

fig = dict( data=data, layout=layout )

py.iplot(fig, filename='whitePizza.html')
wp_df['menus.description'].value_counts()
# Checking the amount of vegetarian / vegan pizzas

print("Quantity: ", len(pizzas[pizzas["menus.name"].str.lower().str.contains("veg")]))
# Number of Vegetarian and Vegan Pizzas

print("Vegetarian Pizzas", len(pizzas[pizzas["menus.name"].str.lower().str.contains("vegetarian")])) 

print("Veggie Pizzas", len(pizzas[pizzas["menus.name"].str.lower().str.contains("veggie")])) 

print("Vegetable Pizzas", len(pizzas[pizzas["menus.name"].str.lower().str.contains("vegetable")])) 
# Gluten Free Pizzas 

pizzas[pizzas["menus.name"].str.lower().str.contains("gluten free")]["menus.name"]
# Vegetarian Pizza

vege_df = pizzas[pizzas['menus.name']=='Vegetarian Pizza'] #Dataframe apenas para Vegetarian Pizza



# Mapa parameters

data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = vege_df['longitude'],

        lat = vege_df['latitude'],

        text = vege_df['city'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.6,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            )))]



# Layout

layout = dict(

        title = 'Locations of Vegetarian Pizzas',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa'),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5))



py.init_notebook_mode(connected=True)

fig = dict( data=data, layout=layout )

py.iplot(fig, filename='vegetarianPizza.html')
# Veggie Pizza

vegg_df = pizzas[pizzas['menus.name']=='Veggie Pizza'] #Dataframe only for Veggie Pizza



# Map Parameters

data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = vegg_df['longitude'],

        lat = vegg_df['latitude'],

        text = vegg_df['city'],

        mode = 'markers',

        marker = dict( 

            size = 8, 

            opacity = 0.6,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            )))]



# Layout

layout = dict(

        title = 'Locations of Veggie Flavor Pizzas',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa'),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5))



py.init_notebook_mode(connected=True)

fig = dict( data=data, layout=layout )

py.iplot(fig, filename='vegetarianPizza.html')
# Cities with more pizzas

pizzas_city = pizzas['city'].value_counts()



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

pizzas_city[:10].plot.bar()

plt.title('Cities with more flavors of pizzas')

plt.ylabel('Quantity')      

plt.show();
qtd_pizzasPhi = len(pd.Series.unique(pizzas[pizzas["city"].str.lower().str.contains("philadelphia")]["menus.name"]))



print('Philadelphia has {} pizzas options!'.format(qtd_pizzasPhi))
# Most Popular Pizzas in Philadelphia

pizzasPhi = pizzas[pizzas["city"].str.lower().str.contains("philadelphia")]["menus.name"].value_counts()



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

pizzasPhi[:10].plot.bar()

plt.title('Flavors with more options in Philadelphia')

plt.ylabel('Quantity')      

plt.show();
# States with more pizza options

pizzas_prov = pizzas['province'].value_counts()



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

pizzas_prov[:30].plot.bar()

plt.title('States with more pizza options')

plt.ylabel('Quantity')      

plt.show();
# Checking non-acronyms

pizzas[pizzas['province'].str.len() > 2]['province']
# Function to change the names of the States

def altera_estado():

    for c in pizzas['province']:

        if c == 'Brentwood':

            pizzas['province'].replace('Brentwood', 'CA', inplace=True)

        if c == 'Los Feliz':

            pizzas['province'].replace('Los Feliz', 'LA', inplace=True)

        if c == 'Ontario Street':

            pizzas['province'].replace('Ontario Street', 'OR', inplace=True)

        if c == 'Manhattan':

            pizzas['province'].replace('Manhattan', 'NY', inplace=True)

        if c == 'Honey Creek':

            pizzas['province'].replace('Honey Creek', 'IN', inplace=True)

        if c == 'Davidsburg':

            pizzas['province'].replace('Davidsburg', 'PA', inplace=True)

        if c == 'Lawrenceville':

            pizzas['province'].replace('Lawrenceville', 'NJ', inplace=True)

        if c == 'Weirs Beach':

            pizzas['province'].replace('Weirs Beach', 'NH', inplace=True)

        if c == 'Dania Beach':

            pizzas['province'].replace('Dania Beach', 'FL', inplace=True)

        if c == 'Fair Haven':

            pizzas['province'].replace('Fair Haven', 'NJ', inplace=True)

        if c == 'Seabrook Island':

            pizzas['province'].replace('Seabrook Island', 'SC', inplace=True)

        if c == 'Valleyview':

            pizzas['province'].replace('Valleyview', 'OH', inplace=True)

        if c == 'Pembroke Pnes':

            pizzas['province'].replace('Pembroke Pnes', 'FL', inplace=True)

        if c == 'Wilm':

            pizzas['province'].replace('Wilm', 'NC', inplace=True)

        if c == 'Hassan':

            pizzas['province'].replace('Hassan', 'MN', inplace=True)

        if c == 'Crestview Heights':

            pizzas['province'].replace('Crestview Heights', 'OR', inplace=True)

        if c == 'Grove':

            pizzas['province'].replace('Grove', 'NJ', inplace=True)

        if c == 'Village Of Wellington':

            pizzas['province'].replace('Village Of Wellington', 'FL', inplace=True)

        if c == 'Oldtown':

            pizzas['province'].replace('Oldtown', 'ID', inplace=True)

        if c == 'Elmhurst':

            pizzas['province'].replace('Elmhurst', 'IL', inplace=True)

        if c == 'Bloomfld Hls':

            pizzas['province'].replace('Bloomfld Hls', 'MI', inplace=True)

        if c == 'Joppatowne':

            pizzas['province'].replace('Joppatowne', 'MD', inplace=True)

        if c == 'Baxter Estates':

            pizzas['province'].replace('Baxter Estates', 'NY', inplace=True)

        if c == 'Co Spgs':

            pizzas['province'].replace('Co Spgs', 'CO', inplace=True)

        if c == 'Queens':

            pizzas['province'].replace('Queens', 'NY', inplace=True)

        if c == 'Midtown':

            pizzas['province'].replace('Midtown', 'NY', inplace=True)

        if c == 'Townley':

            pizzas['province'].replace('Townley', 'AL', inplace=True)

        if c == 'Rivervale':

            pizzas['province'].replace('Rivervale', 'NJ', inplace=True)

        if c == 'Rivervale':

            pizzas['province'].replace('Rivervale', 'NJ', inplace=True)

        if c == 'Pitt':

            pizzas['province'].replace('Pitt', 'PA', inplace=True)

        if c == 'Nyc':

            pizzas['province'].replace('Nyc', 'NY', inplace=True)

        if c == 'Weymouth Nas':

            pizzas['province'].replace('Weymouth Nas', 'MA', inplace=True)

        if c == 'East Htfd':

            pizzas['province'].replace('East Htfd', 'CT', inplace=True)

        if c == 'Rockville':

            pizzas['province'].replace('Rockville', 'MD', inplace=True)

        if c == 'No Bethesda':

            pizzas['province'].replace('No Bethesda', 'MD', inplace=True)

        if c == 'Elnora':

            pizzas['province'].replace('Elnora', 'IN', inplace=True)

        if c == 'Arco-plaza':

            pizzas['province'].replace('Arco-plaza', 'LA', inplace=True)

        if c == 'Brownhelm':

            pizzas['province'].replace('Brownhelm', 'OH', inplace=True)

        if c == 'Bellefonte':

            pizzas['province'].replace('Bellefonte', 'PA', inplace=True)

        if c == 'Bloomington Heights':

            pizzas['province'].replace('Bloomington Heights', 'IL', inplace=True)

        if c == 'West Deerfield':

            pizzas['province'].replace('West Deerfield', 'FL', inplace=True)

        if c == 'West Medford':

            pizzas['province'].replace('West Medford', 'MA', inplace=True)

        if c == 'West Mifflin':

            pizzas['province'].replace('West Mifflin', 'PA', inplace=True)

        if c == 'West Pittsburg':

            pizzas['province'].replace('West Pittsburg', 'PA', inplace=True)

        if c == 'West Vail':

            pizzas['province'].replace('West Vail', 'CO', inplace=True)

        if c == 'Wheatfield':

            pizzas['province'].replace('Wheatfield', 'NY', inplace=True)

        if c == 'Williams Crk':

            pizzas['province'].replace('Williams Crk', 'IN', inplace=True)

        if c == 'Willoughby Hills':

            pizzas['province'].replace('Willoughby Hills', 'OH', inplace=True)

        if c == 'Woodbury':

            pizzas['province'].replace('Woodbury', 'MN', inplace=True)

        if c == 'Burlngtn City':

            pizzas['province'].replace('Burlngtn City', 'VT', inplace=True)

        if c == 'Bunker Hill Village':

            pizzas['province'].replace('Bunker Hill Village', 'TX', inplace=True)

        if c == 'Brownstown Twp':

            pizzas['province'].replace('Brownstown Twp', 'MI', inplace=True)

        if c == 'Brownsboro Farm':

            pizzas['province'].replace('Brownsboro Farm', 'KY', inplace=True)

        if c == 'Briarcliff Mnr':

            pizzas['province'].replace('Briarcliff Mnr', 'NY', inplace=True)

        if c == 'Brandtsville':

            pizzas['province'].replace('Brandtsville', 'PA', inplace=True)

        if c == 'Bouquet Canyon':

            pizzas['province'].replace('Bouquet Canyon', 'CA', inplace=True)

        if c == 'Bonney Lk':

            pizzas['province'].replace('Bonney Lk', 'WA', inplace=True)

        if c == 'Blue Anchor':

            pizzas['province'].replace('Blue Anchor', 'NJ', inplace=True)

        if c == 'Bloomington Hills':

            pizzas['province'].replace('Bloomington Hills', 'UT', inplace=True)

        if c == 'Clarkson Valley':

            pizzas['province'].replace('Clarkson Valley', 'MO', inplace=True)

        if c == 'Colorado Spgs':

            pizzas['province'].replace('Colorado Spgs', 'CO', inplace=True)

        if c == 'Country Life Acres':

            pizzas['province'].replace('Country Life Acres', 'MO', inplace=True)

        if c == 'N Egremont':

            pizzas['province'].replace('N Egremont', 'MA', inplace=True)

        if c == 'North Glenn':

            pizzas['province'].replace('North Glenn', 'CO', inplace=True)

        if c == 'Friendsville':

            pizzas['province'].replace('Friendsville', 'IL', inplace=True)

        if c == 'Groesbeck':

            pizzas['province'].replace('Groesbeck', 'OH', inplace=True)

        if c == 'City Of Spokane Valley':

            pizzas['province'].replace('City Of Spokane Valley', 'WA', inplace=True)

        if c == 'Oella':

            pizzas['province'].replace('Oella', 'MD', inplace=True)

        if c == 'No Providence':

            pizzas['province'].replace('No Providence', 'Providence', inplace=True)   

        if c == 'Murdock':

            pizzas['province'].replace('Murdock', 'MN', inplace=True)   

        if c == 'Quincy Center':

            pizzas['province'].replace('Quincy Center', 'MA', inplace=True)   

        if c == 'Marble Cliff':

            pizzas['province'].replace('Marble Cliff', 'OH', inplace=True)   

        if c == 'New York City':

            pizzas['province'].replace('New York City', 'NY', inplace=True)   

        if c == 'S Connelsvl':

            pizzas['province'].replace('S Connelsvl', 'PA', inplace=True)   

        if c == 'Juanita':

            pizzas['province'].replace('Juanita', 'WA', inplace=True)   

        if c == 'Forest View':

            pizzas['province'].replace('Forest View', 'IL', inplace=True)   

        if c == 'Kingsgate':

            pizzas['province'].replace('Kingsgate', 'WA', inplace=True)   

        if c == 'Mount Laurel Township':

            pizzas['province'].replace('Mount Laurel Township', 'NJ', inplace=True)   

        if c == 'St Albans':

            pizzas['province'].replace('St Albans', 'VT', inplace=True)   

        if c == 'Queensgate':

            pizzas['province'].replace('Queensgate', 'OH', inplace=True)   

        if c == 'Raugust':

            pizzas['province'].replace('Raugust', 'WA', inplace=True)   

        if c == 'Fairmont':

            pizzas['province'].replace('Fairmont', 'WV', inplace=True)   

        if c == 'Providence':

            pizzas['province'].replace('Providence', 'RI', inplace=True)   

        if c == 'Matthewstown':

            pizzas['province'].replace('Matthewstown', 'NC', inplace=True)   

        if c == 'Deephaven':

            pizzas['province'].replace('Deephaven', 'MN', inplace=True)   

        if c == 'Macomb Twp':

            pizzas['province'].replace('Macomb Twp', 'IL', inplace=True)   

        if c == 'Murfreesbr':

            pizzas['province'].replace('Murfreesbr', 'TN', inplace=True)   

        if c == 'Fort Dearborn':

            pizzas['province'].replace('Fort Dearborn', 'IL', inplace=True)   

        if c == 'Saint Davids':

            pizzas['province'].replace('Saint Davids', 'PA', inplace=True)   

        if c == 'Wesley Chapel':

            pizzas['province'].replace('Wesley Chapel', 'FL', inplace=True)   

        if c == 'West Glenville':

            pizzas['province'].replace('West Glenville', 'NY', inplace=True)   

        if c == 'Margate':

            pizzas['province'].replace('Margate', 'FL', inplace=True)   

        if c == 'Miami':

            pizzas['province'].replace('Miami', 'FL', inplace=True)   

        if c == 'Hollywood Park':

            pizzas['province'].replace('Hollywood Park', 'TX', inplace=True)   

        if c == 'East Haven':

            pizzas['province'].replace('East Haven', 'CT', inplace=True)   

        if c == 'Carpolis':

            pizzas['province'].replace('Carpolis', 'LA', inplace=True)   

        if c == 'Cherry Hill Township':

            pizzas['province'].replace('Cherry Hill Township', 'NJ', inplace=True)           

        if c == 'Guilford Courthouse National':

            pizzas['province'].replace('Guilford Courthouse National', 'NC', inplace=True)           
altera_estado()
pizzas_prov = pizzas['province'].value_counts()



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

pizzas_prov[:30].plot.bar()

plt.title('States with more pizza options')

plt.ylabel('Quantity')      

plt.show();
# Initial analysis of values 0 or NA



print("number of zeros in 'priceRangeMin': ", len(pizzas[pizzas['priceRangeMin'] == 0]['priceRangeMin']))

print("number of N/A in 'priceRangeMin': ", pizzas['priceRangeMin'].isnull().sum()) # We had already done the previous increase with the average



print("number of zeros in 'priceRangeMax': ", len(pizzas[pizzas['priceRangeMax'] == 0]['priceRangeMin']))

print("number of N/A in 'priceRangeMax': ", pizzas['priceRangeMax'].isnull().sum()) # We had already done the previous increase with the average
# Increase the valueRangeMin 0 values to the mean

pizzas['priceRangeMin'].replace(pizzas[pizzas['priceRangeMin'] == 0]['priceRangeMin'], round(pizzas['priceRangeMin'].mean(), 2), inplace=True)
print("num of zeros in 'priceRangeMin': ", len(pizzas[pizzas['priceRangeMin'] == 0]['priceRangeMin']))
# Province by Minimum Price



pizzas_provPrice = pizzas[pizzas['province'].str.len() == 2].groupby(['province'])['priceRangeMin'].mean()



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = pizzas_provPrice[:10].plot.bar()

ax.set_title("Minimum Pizza Price per State")

ax.set_ylabel("Value")



rects = ax.patches



for rect in rects:

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

            #'%d' % int(height),

            'R${:1.2f}'.format(height),

            ha='center', va='bottom')



plt.show();
# Province by Maximum Price



pizzas_provPrice = pizzas[pizzas['province'].str.len() == 2].groupby(['province'])['priceRangeMax'].mean()



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = pizzas_provPrice[:10].plot.bar()

ax.set_title("Maximum Pizza Price per State")

ax.set_ylabel("Value")



rects = ax.patches



for rect in rects:

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

            #'%d' % int(height),

            'R${:1.2f}'.format(height),

            ha='center', va='bottom')



plt.show();