# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

import datetime

from plotly.subplots import make_subplots

df = pd.read_excel('/kaggle/input/Data_Extract_From_World_Development_Indicators.xlsx')

df.head()
new_dataframe = df.rename(columns={'Country Name':'Country',

                                  'Country Code':'Code',

                                   '1990 [YR1990]':'1990',

                                  '2000 [YR2000]':'2000',

                                  '2010 [YR2010]':'2010',

                                  '2011 [YR2011]':'2011',

                                  '2012 [YR2012]':'2012',

                                  '2013 [YR2013]':'2013',

                                  '2014 [YR2014]':'2014',

                                  '2015 [YR2015]':'2015',

                                  '2016 [YR2016]':'2016',

                                  '2017 [YR2017]':'2017',

                                  '2018 [YR2018]':'2018',

                                  '2019 [YR2019]':'2019'})

new_dataframe
val = new_dataframe.iloc[0][4:]

df1_frame = pd.DataFrame(val.reset_index())

df1_frame.columns=['Years','Population']

df1_frame
fig = go.Figure()

fig.add_trace(go.Scatter(x=df1_frame['Years'], y=df1_frame['Population'],

                    mode='lines',

                    name='Population'))





fig.update_layout(

    title='Population Report of Nepal after 1990',

    xaxis_title="Years",

    yaxis_title="Population",

    template='plotly_dark'



)



fig.show()
pg_annual = new_dataframe.iloc[1][4:]

pg_annual_df = pd.DataFrame(pg_annual.reset_index())

pg_annual_df.columns=['Years','Population Growth Rate']

pg_annual_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=pg_annual_df['Years'], y=pg_annual_df['Population Growth Rate'],

                    mode='lines+markers',

                    name='Population growth rate',

                        marker_color='green'))





fig.update_layout(

    title='Population growth rate (annual%) of Nepal after 1990',

    xaxis_title="Years",

    yaxis_title="Population growth rate",

    template='plotly_dark'



)



fig.show()
pd_annual = new_dataframe.iloc[3][4:]

pd_annual_df = pd.DataFrame(pd_annual.reset_index())

pd_annual_df.columns=['Years','Population Density']

pd_annual_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=pd_annual_df['Years'], y=pd_annual_df['Population Density'],

                    mode='lines',

                    name='Population density',

                        marker_color='red'))





fig.update_layout(

    title='Population density (people per sq. km of land area)',

    xaxis_title="Years",

    yaxis_title="Population density",

    template='plotly_dark'



)



fig.show()
gni_annual = new_dataframe.iloc[6][4:]

gni_annual_df = pd.DataFrame(gni_annual.reset_index())

gni_annual_df.columns=['Years','Gross National Income']

gni_annual_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gni_annual_df['Years'], y=gni_annual_df['Gross National Income'],

                    mode='lines',

                    name='Currency',

                        marker_color='orange'))





fig.update_layout(

    title='Gross National Income(GNI), Atlas method (current US$)',

    xaxis_title="Years",

    yaxis_title="Gross National Income",

    template='plotly_dark'



)



fig.show()
gni_at_annual = new_dataframe.iloc[7][4:]

gni_at_annual_df = pd.DataFrame(gni_at_annual.reset_index())

gni_at_annual_df.columns=['Years','GNI per capita']

gni_at_annual_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gni_at_annual_df['Years'], y=gni_at_annual_df['GNI per capita'],

                    mode='lines',

                    name='GNI per capita',

                        marker_color='cyan'))





fig.update_layout(

    title='GNI per capita, Atlas method (current US$)	',

    xaxis_title="Years",

    yaxis_title="GNI per capita",

    template='plotly_dark'



)



fig.show()
gni_ppp_annual = new_dataframe.iloc[8][4:]

gni_ppp_annual_df = pd.DataFrame(gni_ppp_annual.reset_index())

gni_ppp_annual_df.columns=['Years','GNI PPP']

gni_ppp_annual_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gni_ppp_annual_df['Years'], y=gni_ppp_annual_df['GNI PPP'],

                    mode='lines',

                    name='GNI PPP',

                        marker_color='lime'))





fig.update_layout(

    title='Gross National Income,Purchasing Power Parity (current international $)',

    xaxis_title="Years",

    yaxis_title="GNI Purchasing Power Parity",

    template='plotly_dark'



)



fig.show()
gni_per_ppp_annual = new_dataframe.iloc[9][4:]

gni_per_ppp_annual_df = pd.DataFrame(gni_per_ppp_annual.reset_index())

gni_per_ppp_annual_df.columns=['Years','GNI per capita PPP']

gni_per_ppp_annual_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gni_per_ppp_annual_df['Years'], y=gni_per_ppp_annual_df['GNI per capita PPP'],

                    mode='lines',

                    name='GNI per capita PPP',

                        marker_color='aqua'))





fig.update_layout(

    title='Gross National Income per capita, Purchasing Power Parity (current international $)	',

    xaxis_title="Years",

    yaxis_title="GNI per Purchasing Power Parity",

    template='plotly_dark'



)



fig.show()
life_exp = new_dataframe.iloc[11][4:]

life_exp_df = pd.DataFrame(life_exp.reset_index())

life_exp_df.columns=['Years','Life Expectancy']

life_exp_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=life_exp_df['Years'], y=life_exp_df['Life Expectancy'],

                         mode='lines',

                        name='Life Expectancy',

                        marker_color='tan'))





fig.update_layout(

    title='Life expectancy at birth, total (years)',

    xaxis_title="Years",

    yaxis_title="Life expectancy at birth",

    template='plotly_dark'



)



fig.show()
fertility_rate = new_dataframe.iloc[12][4:]

fertility_rate_df = pd.DataFrame(fertility_rate.reset_index())

fertility_rate_df.columns=['Years','Fertility Rate']

fertility_rate_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=fertility_rate_df['Years'], y=fertility_rate_df['Fertility Rate'],

                         mode='lines',

                    name='Fertility Rate',

                        marker_color='blue'))





fig.update_layout(

    title='Fertility rate, total (births per woman)',

    xaxis_title="Years",

    yaxis_title="Fertility rate",

    template='plotly_dark'



)



fig.show()
fertility_rate_adol = new_dataframe.iloc[13][4:]

fertility_rate_adol_df = pd.DataFrame(fertility_rate_adol.reset_index())

fertility_rate_adol_df.columns=['Years','Adolescent Fertility Rate']

fertility_rate_adol_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=fertility_rate_adol_df['Years'], y=fertility_rate_adol_df['Adolescent Fertility Rate'],

                         mode='lines',

                    name='Adolescent Fertility Rate',

                        marker_color='coral'))





fig.update_layout(

    title='Adolescent fertility rate (births per 1,000 women ages 15-19)',

    xaxis_title="Years",

    yaxis_title="Adolescent Fertility rate",

    template='plotly_dark'



)



fig.show()
cp = new_dataframe.iloc[16][4:]

cp_df = pd.DataFrame(cp.reset_index())

cp_df.columns=['Years','Mortality Rate']

cp_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=cp_df['Years'], y=cp_df['Mortality Rate'],

                         mode='lines',

                    name='Mortality Rate',

                        marker_color='teal'))





fig.update_layout(

    title='Mortality rate, under-5 (per 1,000 live births)',

    xaxis_title="Years",

    yaxis_title="Mortality Rate",

    template='plotly_dark'



)



fig.show()
im = new_dataframe.iloc[18][4:]

im_df = pd.DataFrame(im.reset_index())

im_df.columns=['Years','Immunization']

im_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=im_df['Years'], y=im_df['Immunization'],

                         mode='lines',

                    name='Immunization',

                        marker_color='darkmagenta'))





fig.update_layout(

    title='Immunization, measles (% of children ages 12-23 months)',

    xaxis_title="Years",

    yaxis_title="Immunization rate",

    template='plotly_dark'



)



fig.show()
pc = new_dataframe.iloc[20][4:]

pc_df = pd.DataFrame(pc.reset_index())

pc_df.columns=['Years','School Enrollment']

pc_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=pc_df['Years'], y=pc_df['School Enrollment'],

                         mode='lines',

                    name='School Enrollment',

                        marker_color='darkgreen'))





fig.update_layout(

    title='School enrollment, primary (% gross)',

    xaxis_title="Years",

    yaxis_title="School Enrollment",

    template='plotly_dark'



)



fig.show()
se = new_dataframe.iloc[21][4:]

se_df = pd.DataFrame(se.reset_index())

se_df.columns=['Years','School Enrollment']

se_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=se_df['Years'], y=se_df['School Enrollment'],

                         mode='lines',

                    name='School Enrollment',

                        marker_color='orange'))





fig.update_layout(

    title='School enrollment, secondary (% gross)',

    xaxis_title="Years",

    yaxis_title="School Enrollment",

    template='plotly_dark'



)



fig.show()
se1 = new_dataframe.iloc[22][4:]

se1_df = pd.DataFrame(se1.reset_index())

se1_df.columns=['Years','School Enrollment']

se1_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=se_df['Years'], y=se_df['School Enrollment'],

                         mode='lines',

                    name='GPI',

                        marker_color='royalblue'))





fig.update_layout(

    title='School enrollment, primary and secondary (gross), gender parity index (GPI)',

    xaxis_title="Years",

    yaxis_title="Enrollment in GPI",

    template='plotly_dark'



)



fig.show()
hiv = new_dataframe.iloc[23][4:]

hiv_df = pd.DataFrame(hiv.reset_index())

hiv_df.columns=['Years','Prevalence of HIV']

hiv_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=hiv_df['Years'], y=hiv_df['Prevalence of HIV'],

                         mode='lines',

                    name='HIV',

                        marker_color='saddlebrown'))





fig.update_layout(

    title='Prevalence of HIV, total (% of population ages 15-49)',

    xaxis_title="Years",

    yaxis_title="HIV",

    template='plotly_dark'



)



fig.show()
forest = new_dataframe.iloc[24][4:]

forest_df = pd.DataFrame(forest.reset_index())

forest_df.columns=['Years','Forest Area']

forest_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=forest_df['Years'], y=forest_df['Forest Area'],

                         mode='lines',

                    name='Forests',

                        marker_color='lightgreen'))





fig.update_layout(

    title='Forest area (sq. km)',

    xaxis_title="Years",

    yaxis_title="Forest Area",

    template='plotly_dark'



)



fig.show()
pop = new_dataframe.iloc[27][4:]

pop_df = pd.DataFrame(pop.reset_index())

pop_df.columns=['Years','Urban Population Growth']

pop_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=pop_df['Years'], y=pop_df['Urban Population Growth'],

                         mode='lines',

                    name='Population',

                        marker_color='navy'))





fig.update_layout(

    title='Urban population growth (annual %)',

    xaxis_title="Years",

    yaxis_title="Urban Population",

    template='plotly_dark'



)



fig.show()
erg = new_dataframe.iloc[28][4:]

erg_df = pd.DataFrame(erg.reset_index())

erg_df.columns=['Years','Energy Use']

erg_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=erg_df['Years'], y=erg_df['Energy Use'],

                         mode='lines',

                    name='Energy',

                        marker_color='olive'))





fig.update_layout(

    title='Energy use (kg of oil equivalent per capita)',

    xaxis_title="Years",

    yaxis_title="Energy Consumption",

    template='plotly_dark'



)



fig.show()
co = new_dataframe.iloc[29][4:]

co_df = pd.DataFrame(co.reset_index())

co_df.columns=['Years','Co2 Emission']

co_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=co_df['Years'], y=co_df['Co2 Emission'],

                         mode='lines',

                    name='co2',

                        marker_color='crimson'))





fig.update_layout(

    title='CO2 emissions (metric tons per capita)',

    xaxis_title="Years",

    yaxis_title="Emission of CO2",

    template='plotly_dark'



)



fig.show()
ep = new_dataframe.iloc[30][4:]

ep_df = pd.DataFrame(ep.reset_index())

ep_df.columns=['Years','Electric Power Consumption']

ep_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=ep_df['Years'], y=ep_df['Electric Power Consumption'],

                         mode='lines',

                    name='Electric power',

                        marker_color='salmon'))





fig.update_layout(

    title='Electric power consumption (kWh per capita)',

    xaxis_title="Years",

    yaxis_title="ELectric power consumption",

    template='plotly_dark'



)



fig.show()
gdp = new_dataframe.iloc[31][4:]

gdp_df = pd.DataFrame(gdp.reset_index())

gdp_df.columns=['Years','GDP']

gdp_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gdp_df['Years'], y=gdp_df['GDP'],

                         mode='lines',

                    name='Gross Domestic Product',

                        marker_color='dodgerblue'))





fig.update_layout(

    title='Gross Domestic Product(GDP) (current US$)',

    xaxis_title="Years",

    yaxis_title="Gross Domestic Product",

    template='plotly_dark'



)



fig.show()
gdp1 = new_dataframe.iloc[32][4:]

gdp1_df = pd.DataFrame(gdp1.reset_index())

gdp1_df.columns=['Years','GDP Growth']

gdp1_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gdp1_df['Years'], y=gdp1_df['GDP Growth'],

                         mode='lines',

                    name='Growth',

                        marker_color='khaki'))





fig.update_layout(

    title='Gross Domestic Product(GDP) Growth (annual%)',

    xaxis_title="Years",

    yaxis_title="Gross Domestic Product Growth",

    template='plotly_dark'



)



fig.show()
gdp2 = new_dataframe.iloc[33][4:]

gdp2_df = pd.DataFrame(gdp2.reset_index())

gdp2_df.columns=['Years','Inflation']

gdp2_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gdp2_df['Years'], y=gdp2_df['Inflation'],

                         mode='lines',

                    name='Inflation',

                        marker_color='indigo'))





fig.update_layout(

    title='Inflation, Gross Domestic Product(GDP) deflator (annual %)',

    xaxis_title="Years",

    yaxis_title="Inflation",

    template='plotly_dark'



)



fig.show()
gdp3 = new_dataframe.iloc[34][4:]

gdp3_df = pd.DataFrame(gdp3.reset_index())

gdp3_df.columns=['Years','AFF']

gdp3_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gdp3_df['Years'], y=gdp3_df['AFF'],

                         mode='lines',

                    name='AFF',

                        marker_color='red'))





fig.update_layout(

    title='Agriculture, forestry, and fishing, value added (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Aff",

    template='plotly_dark'



)



fig.show()
gdp4 = new_dataframe.iloc[35][4:]

gdp4_df = pd.DataFrame(gdp4.reset_index())

gdp4_df.columns=['Years','Industry']

gdp4_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gdp4_df['Years'], y=gdp4_df['Industry'],

                         mode='lines',

                    name='Industry',

                        marker_color='blue'))





fig.update_layout(

    title='Industry (including construction), value added (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Industries",

    template='plotly_dark'



)



fig.show()
exp = new_dataframe.iloc[36][4:]

exp_df = pd.DataFrame(exp.reset_index())

exp_df.columns=['Years','Export']

exp_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=exp_df['Years'], y=exp_df['Export'],

                         mode='lines',

                    name='Export',

                        marker_color='green'))





fig.update_layout(

    title='Exports of goods and services (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Export of goods",

    template='plotly_dark'



)



fig.show()
imp = new_dataframe.iloc[37][4:]

imp_df = pd.DataFrame(imp.reset_index())

imp_df.columns=['Years','Import']

imp_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=imp_df['Years'], y=imp_df['Import'],

                         mode='lines',

                    name='Import',

                        marker_color='gold'))





fig.update_layout(

    title='Imports of goods and services (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Import of goods",

    template='plotly_dark'



)



fig.show()
gc = new_dataframe.iloc[38][4:]

gc_df = pd.DataFrame(gc.reset_index())

gc_df.columns=['Years','GC formation']

gc_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=gc_df['Years'], y=gc_df['GC formation'],

                         mode='lines',

                    name='Gc',

                        marker_color='silver'))





fig.update_layout(

    title='Gross capital formation (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Gross Capital formation",

    template='plotly_dark'



)



fig.show()
rg = new_dataframe.iloc[39][4:]

rg_df = pd.DataFrame(rg.reset_index())

rg_df.columns=['Years','Revenue']

rg_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=rg_df['Years'], y=rg_df['Revenue'],

                         mode='lines',

                    name='',

                        marker_color='maroon'))





fig.update_layout(

    title='Revenue, excluding grants (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Revenue",

    template='plotly_dark'



)



fig.show()
tr = new_dataframe.iloc[40][4:]

tr_df = pd.DataFrame(tr.reset_index())

tr_df.columns=['Years','Time Required']

tr_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=tr_df['Years'], y=tr_df['Time Required'],

                         mode='lines',

                    name='',

                        marker_color='pink'))





fig.update_layout(

    title='Time required to start a business (days)',

    xaxis_title="Years",

    yaxis_title="Time in days",

    template='plotly_dark'



)



fig.show()
tx = new_dataframe.iloc[42][4:]

tx_df = pd.DataFrame(tx.reset_index())

tx_df.columns=['Years','Tax']

tx_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=tx_df['Years'], y=tx_df['Tax'],

                         mode='lines',

                    name='',

                        marker_color='cyan'))





fig.update_layout(

    title='Tax revenue (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Tax Revenue",

    template='plotly_dark'



)



fig.show()
me = new_dataframe.iloc[43][4:]

me_df = pd.DataFrame(me.reset_index())

me_df.columns=['Years','Expenditure']

me_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=me_df['Years'], y=me_df['Expenditure'],

                         mode='lines',

                    name='',

                        marker_color='darkgreen'))





fig.update_layout(

    title='Military expenditure (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Millitary Expenditure",

    template='plotly_dark'



)



fig.show()
mes = new_dataframe.iloc[44][4:]

mes_df = pd.DataFrame(mes.reset_index())

mes_df.columns=['Years','Mobile']

mes_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=mes_df['Years'], y=mes_df['Mobile'],

                         mode='lines',

                    name='',

                        marker_color='darkblue'))





fig.update_layout(

    title='Mobile cellular subscriptions (per 100 people)',

    xaxis_title="Years",

    yaxis_title="Mobile cellular subscriptions",

    template='plotly_dark'



)



fig.show()
mt = new_dataframe.iloc[47][4:]

mt_df = pd.DataFrame(mt.reset_index())

mt_df.columns=['Years','Merchandise']

mt_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=mt_df['Years'], y=mt_df['Merchandise'],

                         mode='lines',

                    name='',

                        marker_color='yellow'))





fig.update_layout(

    title='Merchandise trade (% of GDP)',

    xaxis_title="Years",

    yaxis_title="Merchandise trade",

    template='plotly_dark'



)



fig.show()
eds = new_dataframe.iloc[49][4:]

eds_df = pd.DataFrame(eds.reset_index())

eds_df.columns=['Years','EDS']

eds_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=eds_df['Years'], y=eds_df['EDS'],

                         mode='lines',

                    name='',

                        marker_color='purple'))





fig.update_layout(

    title='External debt stocks, total (DOD, current US$)',

    xaxis_title="Years",

    yaxis_title="External debt stocks",

    template='plotly_dark'



)



fig.show()
tds = new_dataframe.iloc[50][4:]

tds_df = pd.DataFrame(tds.reset_index())

tds_df.columns=['Years','TDS']

tds_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=tds_df['Years'], y=tds_df['TDS'],

                         mode='lines',

                    name='',

                        marker_color='white'))





fig.update_layout(

    title='Total debt service (% of exports of goods, services and primary income)',

    xaxis_title="Years",

    yaxis_title="Total debt service",

    template='plotly_dark'



)



fig.show()
fdi = new_dataframe.iloc[53][4:]

fdi_df = pd.DataFrame(fdi.reset_index())

fdi_df.columns=['Years','TDS']

fdi_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=fdi_df['Years'], y=fdi_df['TDS'],

                         mode='lines',

                    name='',

                        marker_color='red'))





fig.update_layout(

    title='Foreign direct investment, net inflows (BoP, current US$)',

    xaxis_title="Years",

    yaxis_title="Foreign direct investment",

    template='plotly_dark'



)



fig.show()
nod = new_dataframe.iloc[54][4:]

nod_df = pd.DataFrame(nod.reset_index())

nod_df.columns=['Years','NOD']

nod_df
fig = go.Figure()

fig.add_trace(go.Scatter(x=nod_df['Years'], y=nod_df['NOD'],

                         mode='lines',

                    name='',

                        marker_color='green'))





fig.update_layout(

    title='Net official development assistance and official aid received (current US$)',

    xaxis_title="Years",

    yaxis_title="Net official development",

    template='plotly_dark'



)



fig.show()