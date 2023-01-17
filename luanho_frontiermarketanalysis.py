import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = '/kaggle/input/WorldDevelopment.csv'

data = pd.read_csv(path)
list_country = ['Argentina','Bahrain','Bangladesh','Benin','Bulgaria','Croatia','Egypt, Arab Rep.','Ecuador','Estonia','Jamaica','Jordan','Kazakhstan','Kenya','Kuwait','Latvia','Lebanon','Morocco','Nigeria','Oman','Pakistan'\

                    ,'Panama','Papua New Guinea','Qatar','Romania','Serbia','Slovenia','Sri Lanka','Tanzania','Togo','Tunisia' ,'Ukraine','Vietnam','Zambia']



frontier_market_population = data.loc[(data['Series Name'] == "Population, total") & (data['Country Name'].isin(list_country))]
import plotly.express as px

frontier_market_population['2018 [YR2018]'] = frontier_market_population['2018 [YR2018]'].astype(int)

frontier_market_population = frontier_market_population.sort_values(by=['2018 [YR2018]'])

fig_bar = px.bar(frontier_market_population, x = 'Country Name', y='2018 [YR2018]', color = 'Country Name')

fig_bar.show()
list_country = ['Argentina','Bahrain','Bangladesh','Benin','Bulgaria','Croatia','Egypt, Arab Rep.','Ecuador','Estonia','Jamaica','Jordan','Kazakhstan','Kenya','Kuwait','Latvia','Lebanon','Morocco','Nigeria','Oman','Pakistan'\

                    ,'Panama','Papua New Guinea','Qatar','Romania','Serbia','Slovenia','Sri Lanka','Tanzania','Togo','Tunisia' ,'Ukraine','Vietnam','Zambia']

frontier_market_surface_area = data.loc[(data['Series Name'] == "Surface area (sq. km)") & (data['Country Name'].isin(list_country))]

frontier_market_surface_area['2018 [YR2018]'] = frontier_market_surface_area['2018 [YR2018]'].astype(float)

frontier_market_surface_area = frontier_market_surface_area.sort_values('2018 [YR2018]')

fig_bar = px.bar(frontier_market_surface_area, x = 'Country Name', y='2018 [YR2018]', color = 'Country Name')

fig_bar.show()


list_country = ['Argentina','Bahrain','Bangladesh','Benin','Bulgaria','Croatia','Egypt, Arab Rep.','Ecuador','Estonia','Jamaica','Jordan','Kazakhstan','Kenya','Kuwait','Latvia','Lebanon','Morocco','Nigeria','Oman','Pakistan'\

                    ,'Panama','Papua New Guinea','Qatar','Romania','Serbia','Slovenia','Sri Lanka','Tanzania','Togo','Tunisia' ,'Ukraine','Vietnam','Zambia']



frontier_market_gdp = data.loc[(data['Series Name'] == "GDP (current US$)") & (data['Country Name'].isin(list_country))]

frontier_market_gdp['2018 [YR2018]'] = frontier_market_gdp['2018 [YR2018]'].astype(float)

frontier_market_gdp = frontier_market_gdp.sort_values('2018 [YR2018]')

fig_bar = px.bar(frontier_market_gdp, x = 'Country Name', y='2018 [YR2018]', color = 'Country Name')

fig_bar.show()
frontier_market_gdp_and_population = frontier_market_gdp[['Country Name','2018 [YR2018]']].sort_values('Country Name')

frontier_market_population = frontier_market_population.sort_values('Country Name')

frontier_market_gdp_and_population['2018 Population'] = frontier_market_population['2018 [YR2018]'].values

frontier_market_gdp_and_population.columns = ['Country Name', '2018 GDP', '2018 Population']

frontier_market_gdp_and_population = frontier_market_gdp_and_population.sort_values('2018 GDP')

fig = px.scatter(frontier_market_gdp_and_population, x="Country Name", y="2018 GDP" , size="2018 Population", color="Country Name",

                 hover_name="Country Name",size_max=80)

fig.show()
frontier_market_gdp_surface_area = frontier_market_gdp[['Country Name','2018 [YR2018]']].sort_values('Country Name')

frontier_market_surface_area = frontier_market_surface_area.sort_values('Country Name')

frontier_market_gdp_surface_area['2018 Surface Area'] = frontier_market_surface_area['2018 [YR2018]'].values

frontier_market_gdp_surface_area.columns = ['Country Name', '2018 GDP', '2018 Surface Area']

frontier_market_gdp_surface_area = frontier_market_gdp_surface_area.sort_values('2018 GDP')

fig = px.scatter(frontier_market_gdp_surface_area, x="Country Name", y="2018 GDP" , size="2018 Surface Area", color="Country Name",

                 hover_name="Country Name",size_max=80)

fig.show()
frontier_market_net_oda_per_capita = data.loc[(data['Series Name'] == "Net ODA received per capita (current US$)") & (data['Country Name'].isin(list_country))]

frontier_market_net_oda_per_capita = frontier_market_net_oda_per_capita.sort_values('Country Name')
frontier_market_net_oda_per_capita = frontier_market_net_oda_per_capita.sort_values('Country Name')

frontier_market_net_oda_per_capita.fillna(0,inplace=True)

frontier_market_population.fillna(0,inplace=True)
years = ['1990 [YR1990]', '1991 [YR1991]', '1992 [YR1992]', '1993 [YR1993]',

       '1994 [YR1994]', '1995 [YR1995]', '1996 [YR1996]', '1997 [YR1997]',

       '1998 [YR1998]', '1999 [YR1999]', '2000 [YR2000]', '2001 [YR2001]',

       '2002 [YR2002]', '2003 [YR2003]', '2004 [YR2004]', '2005 [YR2005]',

       '2006 [YR2006]', '2007 [YR2007]', '2008 [YR2008]', '2009 [YR2009]',

       '2010 [YR2010]', '2011 [YR2011]', '2012 [YR2012]', '2013 [YR2013]',

       '2014 [YR2014]', '2015 [YR2015]', '2016 [YR2016]', '2017 [YR2017]',

       '2018 [YR2018]']



for y in years:

    frontier_market_net_oda_per_capita[y] =  pd.to_numeric(frontier_market_net_oda_per_capita[y], errors='coerce')

    frontier_market_population[y] =  pd.to_numeric(frontier_market_population[y], errors='coerce')



frontier_market_net_oda_per_capita[y].fillna(0,inplace=True)

frontier_market_population[y].fillna(0,inplace=True)

frontier_market_total_oda = frontier_market_net_oda_per_capita[years] * frontier_market_population[years].values
frontier_market_total_oda['total_oda'] = frontier_market_total_oda[years].sum(axis=1)
frontier_market_total_oda['Country Name'] = frontier_market_population['Country Name'].values

frontier_market_total_oda['2018_Population'] = frontier_market_population['2018 [YR2018]'].values

frontier_market_total_oda = frontier_market_total_oda.sort_values('total_oda')

frontier_market_total_oda.fillna(0,inplace=True)

fig = px.scatter(frontier_market_total_oda, x="Country Name", y="total_oda" , size="2018_Population", color="Country Name",

                 hover_name="Country Name",size_max=80)

fig.show()
frontier_market_fdi = data.loc[(data['Series Name'] == "Foreign direct investment, net (BoP, current US$)") & (data['Country Name'].isin(list_country))]

for y in years:

    frontier_market_fdi[y] = pd.to_numeric(frontier_market_fdi[y], errors = 'coerce')

frontier_market_fdi['total_fdi'] = frontier_market_fdi[years].sum(axis=1)

frontier_market_fdi['total_fdi'] = frontier_market_fdi['total_fdi'].map(lambda x: abs(x) if x<0 else -x)

frontier_market_fdi = frontier_market_fdi.sort_values('total_fdi')
fig_bar = px.bar(frontier_market_fdi, x = 'Country Name', y='total_fdi', color = 'Country Name')

fig_bar.show()
frontier_market_2018_gdp = frontier_market_gdp[['Country Name', '2018 [YR2018]']]

frontier_market_2018_gdp.columns = ['Country Name', '2018 GDP']

frontier_market_fdi = frontier_market_fdi.merge(frontier_market_2018_gdp, how='left', on='Country Name')
frontier_market_fdi = frontier_market_fdi[frontier_market_fdi['total_fdi'] > 0]

frontier_market_fdi['2018 GDP'] = frontier_market_fdi['2018 GDP'].astype(float)

fig = px.scatter(frontier_market_fdi, x="Country Name", y="total_fdi" , size="2018 GDP", color="Country Name",

                 hover_name="Country Name",size_max=80)

fig.show()