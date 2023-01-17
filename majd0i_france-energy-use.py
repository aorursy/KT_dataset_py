#Import Libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from pylab import fill_between



%matplotlib inline



#Read Datasets

country = pd.read_csv('../input/Country.csv')

country_notes = pd.read_csv('../input/CountryNotes.csv')

indicators = pd.read_csv('../input/Indicators.csv')

#Stylistic Options

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    

             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    

             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    

             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    

             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  



for i in range(len(tableau20)):    

    r, g, b = tableau20[i]    

    tableau20[i] = (r / 255., g / 255., b / 255.)
# Plot Access Line Chart for Rural and Urban Communities

df_elec_rural_fr = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.ACCS.RU.ZS')]

df_elec_urban_fr = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.ACCS.UR.ZS')]

df_elec_pop_fr = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_rural_fr.Year,df_elec_rural_fr.Value,'o-',label='Rural',color=tableau20[0])

plt.plot(df_elec_urban_fr.Year,df_elec_urban_fr.Value,'o-',label='Urban',color=tableau20[2])

plt.plot(df_elec_pop_fr.Year,df_elec_pop_fr.Value,'o-',label='General',color=tableau20[1])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Population',  fontsize=14)

plt.title('Access to Electricity in France', fontsize=14)



fig.savefig('access_electricity_France.pdf',format='pdf', dpi=300)

fig.savefig('access_electricity_France.png',format='png', dpi=300)
# Plot Access Line Chart for Rural and Urban Communities

df_elec_rural = indicators[(indicators.CountryName=='Algeria')&(indicators.IndicatorCode=='EG.ELC.ACCS.RU.ZS')]

df_elec_urban = indicators[(indicators.CountryName=='Algeria')&(indicators.IndicatorCode=='EG.ELC.ACCS.UR.ZS')]

df_elec_pop = indicators[(indicators.CountryName=='Algeria')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_rural.Year,df_elec_rural.Value,'o-',label='Rural',color=tableau20[0])

plt.plot(df_elec_urban.Year,df_elec_urban.Value,'o-',label='Urban',color=tableau20[2])

plt.plot(df_elec_pop.Year,df_elec_pop.Value,'o-',label='General',color=tableau20[1])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Population',  fontsize=14)

plt.title('Access to Electricity in Algeria', fontsize=14)



fig.savefig('access_electricity_Algeria.pdf',format='pdf', dpi=300)

fig.savefig('access_electricity_Algeria.png',format='png', dpi=300)
# Plot Access Line Chart for Rural and Urban Communities

df_elec_rural = indicators[(indicators.CountryName=='Chad')&(indicators.IndicatorCode=='EG.ELC.ACCS.RU.ZS')]

df_elec_urban = indicators[(indicators.CountryName=='Chad')&(indicators.IndicatorCode=='EG.ELC.ACCS.UR.ZS')]

df_elec_pop = indicators[(indicators.CountryName=='Chad')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_rural.Year,df_elec_rural.Value,'o-',label='Rural',color=tableau20[0])

plt.plot(df_elec_urban.Year,df_elec_urban.Value,'o-',label='Urban',color=tableau20[2])

plt.plot(df_elec_pop.Year,df_elec_pop.Value,'o-',label='General',color=tableau20[1])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Population',  fontsize=14)

plt.title('Access to Electricity in Chad', fontsize=14)



fig.savefig('access_electricity_Chad.pdf',format='pdf', dpi=300)

fig.savefig('access_electricity_Chad.png',format='png', dpi=300)
df_ch_elec_pop = indicators[(indicators.CountryName=='Chad')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_co_elec_pop = indicators[(indicators.CountryName=='Colombia')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_in_elec_pop = indicators[(indicators.CountryName=='Indonesia')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_ma_elec_pop = indicators[(indicators.CountryName=='Malaysia')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_gh_elec_pop = indicators[(indicators.CountryName=='Ghana')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_al_elec_pop = indicators[(indicators.CountryName=='Algeria')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_fr_elec_pop = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]

df_th_elec_pop = indicators[(indicators.CountryName=='Thailand')&(indicators.IndicatorCode=='EG.ELC.ACCS.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_ch_elec_pop.Year,df_ch_elec_pop.Value,'o-',label='Chad',color=tableau20[0])

plt.plot(df_co_elec_pop.Year,df_co_elec_pop.Value,'o-',label='Colombia',color=tableau20[1])

plt.plot(df_in_elec_pop.Year,df_in_elec_pop.Value,'o-',label='Indonesia',color=tableau20[2])

plt.plot(df_ma_elec_pop.Year,df_ma_elec_pop.Value,'o-',label='Malaysia',color=tableau20[3])

plt.plot(df_gh_elec_pop.Year,df_gh_elec_pop.Value,'o-',label='Ghana',color=tableau20[4])

plt.plot(df_al_elec_pop.Year,df_in_elec_pop.Value,'o-',label='Algeria',color=tableau20[5])

plt.plot(df_fr_elec_pop.Year,df_fr_elec_pop.Value,'o-',label='France',color=tableau20[6])

plt.plot(df_th_elec_pop.Year,df_th_elec_pop.Value,'o-',label='Thailand',color=tableau20[7])



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xlabel('Years',  fontsize=10)

plt.ylabel('% of Population',  fontsize=10)

plt.title('Access to Electricity Arround the world', fontsize=10)

plt.ylim([0,110])

fig.savefig('access_electricity_world.pdf',format='pdf', dpi=300)

fig.savefig('access_electricity_world.png',format='png', dpi=300)
df_elec_fosl = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_elec_hydro = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.HYRO.ZS')]

df_elec_nucl = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.NUCL.ZS')]

df_elec_rnwx = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]





width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_fosl.Year,df_elec_fosl.Value,label='Fossil Fuels',color=tableau20[6])

plt.plot(df_elec_hydro.Year,df_elec_hydro.Value,label='Hydroelectric',color=tableau20[0])

plt.plot(df_elec_nucl.Year,df_elec_nucl.Value,label='Nuclear',color=tableau20[3])

plt.plot(df_elec_rnwx.Year,df_elec_rnwx.Value,label='Renewable',color=tableau20[4])





fill_between(df_elec_fosl.Year,df_elec_fosl.Value,0,alpha=0.5,color=tableau20[6])

fill_between(df_elec_hydro.Year,df_elec_hydro.Value,0,alpha=0.5,color=tableau20[0])

fill_between(df_elec_nucl.Year,df_elec_nucl.Value,0,alpha=0.5,color=tableau20[3])

fill_between(df_elec_rnwx.Year,df_elec_rnwx.Value,0,alpha=0.5,color=tableau20[4])

fill_between(df_elec_rnwx.Year,df_elec_rnwx.Value,0,alpha=0.5,color=tableau20[4])

#fill_between(x,y2,0,color='magenta')

#fill_between(x,y3,0,color='red')



plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Total Energy Produce',  fontsize=14)

plt.title('Energy Mix in France (1960-2012)', fontsize=18)





fig.savefig('energy_mix_France.pdf',format='pdf', dpi=300)

fig.savefig('energy_mix_France.png',format='png', dpi=300)
df_elec_fosl_ag = indicators[(indicators.CountryName=='Germany')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_elec_hydro_ag = indicators[(indicators.CountryName=='Germany')&(indicators.IndicatorCode=='EG.ELC.HYRO.ZS')]

df_elec_nucl_ag = indicators[(indicators.CountryName=='Germany')&(indicators.IndicatorCode=='EG.ELC.NUCL.ZS')]

df_elec_rnwx_ag = indicators[(indicators.CountryName=='Germany')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]





width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_fosl_ag.Year,df_elec_fosl_ag.Value,label='Fossil Fuels',color=tableau20[6])

plt.plot(df_elec_hydro_ag.Year,df_elec_hydro_ag.Value,label='Hydroelectric',color=tableau20[0])

plt.plot(df_elec_nucl_ag.Year,df_elec_nucl_ag.Value,label='Nuclear',color=tableau20[3])

plt.plot(df_elec_rnwx_ag.Year,df_elec_rnwx_ag.Value,label='Renewable',color=tableau20[4])





fill_between(df_elec_fosl_ag.Year,df_elec_fosl_ag.Value,0,alpha=0.5,color=tableau20[6])

fill_between(df_elec_hydro_ag.Year,df_elec_nucl_ag.Value,0,alpha=0.5,color=tableau20[0])

fill_between(df_elec_nucl_ag.Year,df_elec_nucl_ag.Value,0,alpha=0.5,color=tableau20[3])

fill_between(df_elec_rnwx_ag.Year,df_elec_rnwx_ag.Value,0,alpha=0.5,color=tableau20[4])

fill_between(df_elec_rnwx_ag.Year,df_elec_rnwx_ag.Value,0,alpha=0.5,color=tableau20[4])

#fill_between(x,y2,0,color='magenta')

#fill_between(x,y3,0,color='red')



plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Total Energy Produce',  fontsize=14)

plt.title('Energy Mix in Germany (1960-2012)', fontsize=18)





fig.savefig('energy_mix_Germany.pdf',format='pdf', dpi=300)

fig.savefig('energy_mix_Germany.png',format='png', dpi=300)
df_elec_fosl_ag = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_elec_hydro_ag = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.HYRO.ZS')]

df_elec_nucl_ag = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.NUCL.ZS')]

df_elec_rnwx_ag = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]





width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_fosl_ag.Year,df_elec_fosl_ag.Value,label='Fossil Fuels',color=tableau20[6])

plt.plot(df_elec_hydro_ag.Year,df_elec_hydro_ag.Value,label='Hydroelectric',color=tableau20[0])

plt.plot(df_elec_nucl_ag.Year,df_elec_nucl_ag.Value,label='Nuclear',color=tableau20[3])

plt.plot(df_elec_rnwx_ag.Year,df_elec_rnwx_ag.Value,label='Renewable',color=tableau20[4])





fill_between(df_elec_fosl_ag.Year,df_elec_fosl_ag.Value,0,alpha=0.5,color=tableau20[6])

fill_between(df_elec_hydro_ag.Year,df_elec_nucl_ag.Value,0,alpha=0.5,color=tableau20[0])

fill_between(df_elec_nucl_ag.Year,df_elec_nucl_ag.Value,0,alpha=0.5,color=tableau20[3])

fill_between(df_elec_rnwx_ag.Year,df_elec_rnwx_ag.Value,0,alpha=0.5,color=tableau20[4])

fill_between(df_elec_rnwx_ag.Year,df_elec_rnwx_ag.Value,0,alpha=0.5,color=tableau20[4])

#fill_between(x,y2,0,color='magenta')

#fill_between(x,y3,0,color='red')



plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Total Energy Produce',  fontsize=14)

plt.title('Energy Mix in Japan (1960-2012)', fontsize=18)





fig.savefig('energy_mix_Japan.pdf',format='pdf', dpi=300)

fig.savefig('energy_mix_Japan.png',format='png', dpi=300)
df_elec_ngas = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.NGAS.ZS')]

df_elec_coal = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.COAL.ZS')]

df_elec_petr = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.ELC.PETR.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_ngas.Year,df_elec_ngas.Value,label='Natural Gas',color=tableau20[9])

plt.plot(df_elec_coal.Year,df_elec_coal.Value,label='Coal',color=tableau20[10])

plt.plot(df_elec_petr.Year,df_elec_petr.Value,label='Petroleum',color=tableau20[11])



fill_between(df_elec_petr.Year,df_elec_petr.Value,0,alpha=0.5,color=tableau20[11])

fill_between(df_elec_coal.Year,df_elec_coal.Value,0,alpha=0.5,color=tableau20[10])

fill_between(df_elec_ngas.Year,df_elec_ngas.Value,0,alpha=0.5,color=tableau20[9])







plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Total Energy Produce',  fontsize=14)

plt.title('Fossil Fuel Mix in France (1960-2012)', fontsize=18)





fig.savefig('fossil_fuel_mix.pdf',format='pdf', dpi=300)

fig.savefig('fossil_fuel_mix.png',format='png', dpi=300)
df_elec_ngas = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.NGAS.ZS')]

df_elec_coal = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.COAL.ZS')]

df_elec_petr = indicators[(indicators.CountryName=='Japan')&(indicators.IndicatorCode=='EG.ELC.PETR.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_ngas.Year,df_elec_ngas.Value,label='Natural Gas',color=tableau20[9])

plt.plot(df_elec_coal.Year,df_elec_coal.Value,label='Coal',color=tableau20[10])

plt.plot(df_elec_petr.Year,df_elec_petr.Value,label='Petroleum',color=tableau20[11])



fill_between(df_elec_petr.Year,df_elec_petr.Value,0,alpha=0.5,color=tableau20[11])

fill_between(df_elec_coal.Year,df_elec_coal.Value,0,alpha=0.5,color=tableau20[10])

fill_between(df_elec_ngas.Year,df_elec_ngas.Value,0,alpha=0.5,color=tableau20[9])







plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Total Energy Produce',  fontsize=14)

plt.title('Fossil Fuel Mix in Japan (1960-2012)', fontsize=18)





fig.savefig('fossil_fuel_mix_Japan.pdf',format='pdf', dpi=300)

fig.savefig('fossil_fuel_mix_Japan.png',format='png', dpi=300)
df_ca_elec_pop = indicators[(indicators.CountryName=='Chad')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_rs_elec_pop = indicators[(indicators.CountryName=='Colombia')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_pk_elec_pop = indicators[(indicators.CountryName=='Indonesia')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_ma_elec_pop = indicators[(indicators.CountryName=='Malaysia')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_in_elec_pop = indicators[(indicators.CountryName=='Ghana')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_ph_elec_pop = indicators[(indicators.CountryName=='Algeria')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_si_elec_pop = indicators[(indicators.CountryName=='Singapore')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]

df_th_elec_pop = indicators[(indicators.CountryName=='Thailand')&(indicators.IndicatorCode=='EG.ELC.FOSL.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_ca_elec_pop.Year,df_ca_elec_pop.Value,label='Chad',color=tableau20[7])

plt.plot(df_rs_elec_pop.Year,df_rs_elec_pop.Value,label='Colombia',color=tableau20[4])

plt.plot(df_pk_elec_pop.Year,df_pk_elec_pop.Value,label='Indonesia',color=tableau20[8])

plt.plot(df_ma_elec_pop.Year,df_ma_elec_pop.Value,label='Malaysia',color=tableau20[10])

plt.plot(df_in_elec_pop.Year,df_in_elec_pop.Value,label='Ghana',color=tableau20[2])

plt.plot(df_ph_elec_pop.Year,df_ph_elec_pop.Value,label='Algeria',color=tableau20[6])

plt.plot(df_si_elec_pop.Year,df_si_elec_pop.Value,label='Singapore',color=tableau20[3])

plt.plot(df_th_elec_pop.Year,df_th_elec_pop.Value,label='Thailand',color=tableau20[5])





plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Energy Production',  fontsize=14)

plt.title('Fossil Fuel Use Arround the world', fontsize=18)



plt.ylim([0,110])

plt.xlim([1990,2019])

fig.savefig('fossil_fuel_electricity_world.pdf',format='pdf', dpi=300)

fig.savefig('fossil_fuel_electricity_world.png',format='png', dpi=300)
df_br_elec_pop = indicators[(indicators.CountryName=='Chad')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_ca_elec_pop = indicators[(indicators.CountryName=='Russia')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_in_elec_pop = indicators[(indicators.CountryName=='Ghana')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_la_elec_pop = indicators[(indicators.CountryName=='Algeria')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_ma_elec_pop = indicators[(indicators.CountryName=='Malaysia')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_in_elec_pop = indicators[(indicators.CountryName=='Colombia')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_ph_elec_pop = indicators[(indicators.CountryName=='Philippines')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_si_elec_pop = indicators[(indicators.CountryName=='Singapore')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_th_elec_pop = indicators[(indicators.CountryName=='Thailand')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_ti_elec_pop = indicators[(indicators.CountryName=='Timor-Leste')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]

df_vi_elec_pop = indicators[(indicators.CountryName=='Vietnam')&(indicators.IndicatorCode=='EG.ELC.RNWX.ZS')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_si_elec_pop.Year,df_si_elec_pop.Value,label='Singapore',color=tableau20[7])

plt.plot(df_ma_elec_pop.Year,df_ma_elec_pop.Value,label='Malaysia',color=tableau20[4])

plt.plot(df_th_elec_pop.Year,df_th_elec_pop.Value,label='Thailand',color=tableau20[8])

plt.plot(df_vi_elec_pop.Year,df_vi_elec_pop.Value,label='Vietnam',color=tableau20[10])

plt.plot(df_pk_elec_pop.Year,df_pk_elec_pop.Value,label='Ghana',color=tableau20[2])

plt.plot(df_ph_elec_pop.Year,df_ph_elec_pop.Value,label='Philippines',color=tableau20[6])

plt.plot(df_la_elec_pop.Year,df_la_elec_pop.Value,label='Algeria',color=tableau20[3])

plt.plot(df_in_elec_pop.Year,df_in_elec_pop.Value,label='Colombia',color=tableau20[5])

plt.plot(df_ca_elec_pop.Year,df_ca_elec_pop.Value,label='Chad',color=tableau20[0])

plt.plot(df_ti_elec_pop.Year,df_ti_elec_pop.Value,label='Timor-Leste',color=tableau20[9])

plt.plot(df_rs_elec_pop.Year,df_rs_elec_pop.Value,label='Russia',color=tableau20[1])





plt.legend(loc=1, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('% of Energy Production',  fontsize=14)

plt.title('Renewable Energy Adoption Arround the world', fontsize=18)



plt.ylim([0,100])

plt.xlim([1990,2019])

fig.savefig('renewable_electricity_world.pdf',format='pdf', dpi=300)

fig.savefig('renewable_electricity_world.png',format='png', dpi=300)
df_elec_use = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EG.USE.ELEC.KH.PC')]





width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_use.Year,df_elec_use.Value,color=tableau20[3])



#plt.legend(loc=4, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('kWh per capita',  fontsize=14)

plt.title('Electric Power Consumption in France', fontsize=18)





fig.savefig('electric_consumption_france.pdf',format='pdf', dpi=300)

fig.savefig('electric_consumption_france.png',format='png', dpi=300)
df_elec_emi = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EN.ATM.CO2E.KT')]

df_elec_gf = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EN.ATM.CO2E.GF.KT')]

df_elec_lf = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EN.ATM.CO2E.LF.KT')]

df_elec_sf = indicators[(indicators.CountryName=='France')&(indicators.IndicatorCode=='EN.ATM.CO2E.SF.KT')]



width = 16

height = 12

fig = plt.figure(figsize=(width, height))



plt.plot(df_elec_emi.Year,df_elec_emi.Value,label='C0$_2$ emissions',color=tableau20[1])

plt.plot(df_elec_lf.Year,df_elec_lf.Value,label='C0$_2$ emissions from liquid fuel',color=tableau20[3])

plt.plot(df_elec_sf.Year,df_elec_sf.Value,label='C0$_2$ emissions from solid fuel',color=tableau20[4])

plt.plot(df_elec_gf.Year,df_elec_gf.Value,label='C0$_2$ emissions from gaseous fuel',color=tableau20[2])



fill_between(df_elec_emi.Year,df_elec_emi.Value,0,alpha=0.5,color=tableau20[1])

fill_between(df_elec_lf.Year,df_elec_lf.Value,0,alpha=0.5,color=tableau20[3])

fill_between(df_elec_sf.Year,df_elec_sf.Value,0,alpha=0.5,color=tableau20[4])

fill_between(df_elec_gf.Year,df_elec_gf.Value,0,alpha=0.5,color=tableau20[2])



plt.legend(loc=2, borderaxespad=1.)

plt.xlabel('Years',  fontsize=14)

plt.ylabel('kt (kilotons)',  fontsize=14)

plt.title('Carbon Footprint in France', fontsize=18)





fig.savefig('co2_emissions_france.pdf',format='pdf', dpi=300)

fig.savefig('co2_emissions_france.png',format='png', dpi=300)