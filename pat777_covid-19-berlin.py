from datetime import datetime

import numpy as np

import pandas as pd

import geopandas as gpd

import geoplot

import mapclassify

import matplotlib.pyplot as plt

import matplotlib.colors as colors



%matplotlib inline

import matplotlib.pyplot as pyplot
HISTORY_DATA_URL='https://funkeinteraktiv.b-cdn.net/history.v4.csv'

#HISTORY_DATA_URL='https://interaktiv.morgenpost.de/data/corona/cases.rki.v2.csv'
START_DATE='05-mar-2020'

LATEST_DATE='13-oct-2020'
df = pd.read_csv(HISTORY_DATA_URL, usecols=['parent', 'label', 'label_parent', 'population', 'date', 'updated', 'confirmed', 'recovered', 'deaths'])

df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")



history_df = df[(df.date>pd.to_datetime(START_DATE)) & (df.date<=LATEST_DATE)]
LATEST_DATE
bln_df = history_df[history_df.label=='Berlin'].copy()

bln_df.drop(columns=['parent','label_parent','updated'],inplace=True)

bln_df['active'] = bln_df['confirmed'] - (bln_df['recovered'] + bln_df['deaths'])
BLN_POPULATION=int(bln_df['population'].tail(1))
de_bln_all = bln_df.drop(columns=['population','label']).set_index('date')
de_bln = history_df[history_df.parent=='de.be'].copy() # does not work any more for Charlottenburg :-(

de_bln.drop(columns=['parent','label_parent','updated'],inplace=True)

de_bln['active'] = de_bln['confirmed'] - (de_bln['recovered'] + de_bln['deaths'])
population_per_district_bln = de_bln[['label','population']].dropna().rename(columns={'label':'district'}).groupby('district').last()
def draw_threshold_lines():

    plt.axhline(30, color='red')

    plt.text(LATEST_DATE,31,'Rote Berliner Ampel',ha='right', color='red')

    plt.axhline(50, color='darkred')

    plt.text(LATEST_DATE,51,'German Maximum',ha='right', color='darkred')

       

def severity_color_berliner_ampel(c):

    if c >= 50:

        return 'background-color: darkred'

    if c >= 30:

        return 'background-color: red'          

    if c >= 20:

        return 'background-color: orange'        

    else:

        return 'background-color: white'    
bln = (

    gpd.read_file('../input/berlin-districts/berliner_bezirke.shp', encoding='utf-8')

        .rename(columns={'name':'district'})[['district','geometry']]

        .set_index('district')

)
de_bln_all.tail()
de_bln_all.plot(figsize=(12,5), title='Berlin: Absolute Cases')

pyplot.grid(True);
def abs_growth_from(df_by_date):

    return df_by_date.diff().fillna(0)



def plot_abs_growth(df_abs_growth ,details="",column='confirmed', max_y=None):

    max_y = max_y if max_y else df_abs_growth[column].max()

    ax = df_abs_growth.plot(figsize=(12,4), title='New {} 1day: {}'.format(column, details), ylim=(0,max_y))

    pyplot.ylabel('Delta to day before');

    pyplot.grid(True)

    return ax
abs_growth_bln_by_date = abs_growth_from(de_bln_all)

abs_growth_bln_by_date.tail()
plot_abs_growth(abs_growth_bln_by_date,'Berlin');
abs_growth_c = abs_growth_bln_by_date['confirmed']

abs_growth_confirmed = pd.DataFrame({'confirmed':abs_growth_c, 'confirmed_mean_7days' : abs_growth_c.rolling(7).mean()})

plot_abs_growth(abs_growth_confirmed,'Berlin');
abs_growth_d = abs_growth_bln_by_date['deaths']

abs_growth_deaths = pd.DataFrame({'deaths':abs_growth_d, 'deaths_mean_7days' : abs_growth_d.rolling(7).mean()})

plot_abs_growth(abs_growth_deaths,'Berlin','deaths');
PER_INHABITANTS = 100000
new_infections7days_bln=abs_growth_c.rolling(7).sum()*PER_INHABITANTS/BLN_POPULATION

new_deaths7days_bln=abs_growth_d.rolling(7).sum()*PER_INHABITANTS/BLN_POPULATION

new_7days_bln=pd.DataFrame({'new_infections_7days_per100k':new_infections7days_bln, 'new_deaths_7days_per100k' : new_deaths7days_bln})
new_7days_bln.tail(7).style.applymap(lambda c: severity_color_berliner_ampel(c))
plt.figure(figsize=(14,7))



ax1 = plt.subplot(121, title = "Berlin: New Infections last 7 days per 100k inhabitants")

new_infections7days_bln.plot(ax=ax1, grid=True)



draw_threshold_lines()



ax2 = plt.subplot(122, title = "Berlin: New Deaths last 7 days per 100k inhabitants")

new_deaths7days_bln.plot(ax=ax2, grid=True);
DAYS_BEFORE=4

DAYS=7

abs_growth_c_lastXdays = abs_growth_c.rolling(DAYS).sum()

r = abs_growth_c_lastXdays/abs_growth_c_lastXdays.shift(4)

r.tail(14)
def ampel_color_r(r):

    r_last3days = r.tail(3)

    if (r_last3days>1.2).sum()>=3:

        return 'red'

    if (r_last3days>1.1).sum()>=3:

        return 'yellow'

    else:

        return 'green'

    

def ampel_color_new_infections7days(new_infections7days):

    last_new = new_infections7days[-1]

    if last_new>=30:

        return 'red'

    if last_new>=20:

        return 'yellow'

    else:

        return 'green'    

    

print ("R-Wert         : {}".format(ampel_color_r(r)))

print ("7-Tage-Incidenz: {}".format(ampel_color_new_infections7days(new_7days_bln.new_infections_7days_per100k)))
bln_districts = de_bln[de_bln.label != 'weitere Fälle in Berlin'].rename(columns={'label':'district'})
bln_districts_by_date_district=bln_districts.set_index(['date','district'])

bln_districts_by_date_district['active_per_inhabitants']=bln_districts_by_date_district['active']/bln_districts_by_date_district['population']*PER_INHABITANTS
bln_district_cases_by_date=bln_districts_by_date_district.reset_index().drop(columns=['population']).set_index(['date']).groupby('district')

bln_district_cases_by_date.last().sort_values(by='active_per_inhabitants', ascending=False)
bln_district_cases_by_date['confirmed'].plot(figsize=(15,7), legend=True, title='Absolute Confirmed by Berlin Districts');

pyplot.grid(True)
bln_district_cases_by_date['active'].plot(figsize=(15,7), legend=True, title='Absolute Active by Berlin Districts');

pyplot.grid(True)
active_per_inhabitants_w_geo=bln.merge(bln_district_cases_by_date.last()[['active_per_inhabitants']], left_index=True, right_index=True)
bln_district_cases_by_date['active_per_inhabitants'].plot(figsize=(15,7), legend=True, title='Active per 100.000 inhabitants by Berlin Districts');

pyplot.grid(True)
geoplot.choropleth(

    active_per_inhabitants_w_geo, 

    hue='active_per_inhabitants',

    cmap='Reds',

    legend=True

);

plt.suptitle('Current active per 100k inhabitants');
bln_districts_new_by_date_and_district=abs_growth_from(bln_districts_by_date_district['confirmed'])

bln_districts_new = bln_districts_new_by_date_and_district.reset_index().rename(columns={'confirmed':'new_infections'})
bln_districts_new_by_date_per_district = pd.pivot_table(bln_districts_new, values='new_infections', index=['date'],

                    columns=['district']).tail(16)
def weekenddays(s):

    return ['background-color: grey' if s.date.dayofweek in [5,6] else '']*s.size



def null_report(c):

    return 'background-color : yellow' if isinstance(c, (int, float, complex))  and c<1 else ''

    

    

bln_districts_new_by_date_per_district.reset_index().style.apply(weekenddays, axis=1).applymap(null_report).hide_index()
bln_districts_new_by_date=bln_districts_new.set_index('date').groupby('district')

bln_districts_new_by_date_last=bln_districts_new_by_date.last()

bln_districts_new_by_date_last['new/max %'] = bln_districts_new_by_date.last()/bln_districts_new_by_date.max()*100



# FIX drop Charlottenburg-Wilmersdorf, because we don't get any data anymore

new_infections = bln_districts_new_by_date_last.drop('Charlottenburg-Wilmersdorf')



new_infections.sort_values(by='new/max %', ascending=False, inplace=True)
def highlight(df):

    num_of_cols = df.size

    if df.new_infections < 1.0:

        return ['background-color: yellow']*num_of_cols

    else:

        if df['new/max %'] >= 100:

            return ['background-color: red']*num_of_cols

        else:

            return ['background-color: white']*num_of_cols



new_infections.style.apply(highlight, axis=1)        
bln_districts_new_by_date_last.new_infections.sum()
bln_districts_new7days = bln_districts_new_by_date_and_district.rolling(7).sum().reset_index().rename(columns={'confirmed':'new_infections7days'}) 



# FIX drop Charlottenburg-Wilmersdorf, because we don't get any data anymore

wo_charlottenburg=bln_districts_new7days.set_index('district').drop('Charlottenburg-Wilmersdorf')

bln_districts_new7days_with_pop = wo_charlottenburg.merge(population_per_district_bln, left_index=True, right_index=True).reset_index()
bln_districts_new7days_by_date_with_pop = bln_districts_new7days_with_pop.set_index('date')
bln_districts_new7days_by_date_with_pop['new_infections7days_per100k'] = (

    bln_districts_new7days_by_date_with_pop['new_infections7days']/bln_districts_new7days_by_date_with_pop['population']*PER_INHABITANTS

)

bln_districts_new7days_by_date=bln_districts_new7days_by_date_with_pop.drop(columns=['population'])
bln_districts_new7days_last=bln_districts_new7days_by_date.groupby(by="district").last()

new7days = bln_districts_new7days_last.sort_values(by='new_infections7days_per100k', ascending=False)
def highlight(df):

    num_of_cols = df.size

    if df.new_infections7days < 1.0:

        return ['background-color: yellow']*num_of_cols

    else:

        return [severity_color_berliner_ampel(df['new_infections7days_per100k'])]*num_of_cols

        

new7days.style.apply(highlight, axis=1)        
bln_districts_new7days_by_date.groupby(by="district")['new_infections7days'].plot(

    figsize=(17,7), legend=True, title='Absolute New Infections 7 days by Berlin Districts',

    ylim=(0,bln_districts_new7days_by_date['new_infections7days'].max())

);

pyplot.grid(True)
new7d_w_geo=bln.merge(bln_districts_new7days_last[['new_infections7days_per100k']], left_index=True, right_index=True)

new7d_w_geo['new_infections7days_per100k_cut30'] = new7d_w_geo['new_infections7days_per100k'].apply(lambda v:v if v<30 else 30)

new7d_w_geo['new_infections7days_per100k_cut50'] = new7d_w_geo['new_infections7days_per100k'].apply(lambda v:v if v<50 else 50)
plt.figure(figsize=(15,7))

ax = plt.subplot(131, title = "New Infections last 7 days per 100k inhabitants")

geoplot.choropleth(

    new7d_w_geo, 

    hue='new_infections7days_per100k',

    cmap='Reds',

    legend=True,

    legend_kwargs={'orientation': 'horizontal'},

    ax=ax

);



if (new7d_w_geo.new_infections7days_per100k <= 30).sum() >0:

    ax = plt.subplot(132, title = "New Infections last 7 days per 100k inhabitants\n cut to max=30 (Berliner Ampel)")

    geoplot.choropleth(

        new7d_w_geo, 

        hue='new_infections7days_per100k_cut30',

        cmap='Reds',

        legend=True,

        legend_kwargs={'orientation': 'horizontal'},    

        ax=ax

    );





ax = plt.subplot(133, title = "New Infections last 7 days per 100k inhabitants\n  cut to max=50 (German Maximum)")

geoplot.choropleth(

    new7d_w_geo, 

    hue='new_infections7days_per100k_cut50',

    cmap='Reds',

    legend=True,

    legend_kwargs={'orientation': 'horizontal'},    

    ax=ax

);
bln_districts_new7days_by_date.groupby(by="district")['new_infections7days_per100k'].plot(

    figsize=(17,7), legend=True, title='New Infections 7 days per 100.000 inhabitants by Berlin Districts',

    ylim=(0,bln_districts_new7days_by_date['new_infections7days_per100k'].max())

);



draw_threshold_lines()

pyplot.grid(True)