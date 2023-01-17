import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



file = '../input/world-renewable-energy-data/World_Renewable_Energy_Data.csv'

data = pd.read_csv(file)

df = pd.DataFrame(data)
df_org = df.copy()

df_org = df_org[df_org['Entity'] == 'World']







col = ['Electricity from Coal (TWh)',

       'Electricity from Gas (TWh)',

       'Electricity from Oil (TWh)',

'Electricity from Hydro (TWh)',

'Electricity from Nuclear (TWh)',

'Electricity from Wind (TWh)',

'Electricity from Solar (TWh)',

'Electricity from Other Renewables (TWh)']



df_org['Electricity Total (TWh)'] = 0



for i in range(len(col)):

    df_org['Electricity Total (TWh)'] = df_org['Electricity Total (TWh)'] + df_org[col[i]]



sizes = [0,0,0,0,0,0,0,0]



for i in range(len(col)):

    sizes[i] = df_org.loc[839,col[i]]/df_org.loc[839,'Electricity Total (TWh)']*100
df_org = df_org.sort_values('Year',ascending=True)

# df_org.set_index('Year', inplace=True)



energy = ['Coal','Gas','Oil','Hydro','Nuclear','Wind','Solar','Other Renewables']



for i in range(len(energy)):

    s = 'Electricity Share from '+ energy[i] + ' (%)'

    df_org[s] = df_org[col[i]]/df_org['Electricity Total (TWh)']

    

filt = df_org['Year'] >= 1985

df_1985 = df_org[filt]
fig = plt.figure(figsize = (15, 10)) 



blah = ['Other Renewables','Solar','Wind','Nuclear','Hydro','Oil','Gas','Coal']



a, b=[plt.cm.Reds, plt.cm.Greens]    



plt.stackplot(df_1985['Year'],df_1985[col[7]],df_1985[col[6]],df_1985[col[5]],df_1985[col[4]],df_1985[col[3]],df_1985[col[2]],df_1985[col[1]],df_1985[col[0]],colors=[b(0.6), b(0.5), b(0.4), b(0.3), b(0.2),a(0.5), a(0.4), a(0.3)])

plt.xlim(1985,2019)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.xlabel('Year',size=15)

plt.ylabel('Electricity Generated (TWh)',size=15)

plt.title('World Electrical Grid Energy Source',size=20)

plt.title('Global Electricity Generation (TWh) by Energy Source',size=20)

plt.legend(blah, prop={"size":13},loc='upper left')

plt.show()
fig = plt.figure(figsize = (15, 10)) 



s = ['Coal','Gas','Oil','Hydro','Nuclear','Wind','Solar','Other Renewables']



# l = energy.reverse() 

# print(l)



for i in range(len(energy)):

    s[i] = 'Electricity Share from '+ s[i] + ' (%)'



a, b=[plt.cm.Reds, plt.cm.Greens]    



plt.stackplot(df_1985['Year'],df_1985[s[7]]*100,df_1985[s[6]]*100,df_1985[s[5]]*100,df_1985[s[4]]*100,df_1985[s[3]]*100,df_1985[s[2]]*100,df_1985[s[1]]*100,df_1985[s[0]]*100,colors=[b(0.6), b(0.5), b(0.4), b(0.3), b(0.2),a(0.5), a(0.4), a(0.3)])

plt.ylim(0,100)

plt.xlim(1985,2019)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.xlabel('Year',size=15)

plt.ylabel('Electricity Generated Share (%)',size=15)

plt.title('Global Electricity Generation Share (%) by Energy Source',size=20)

plt.legend(blah, prop={"size":13},loc='upper left')

plt.show()
# https://python-graph-gallery.com/163-donut-plot-with-subgroups/



group_names=['Fossil Fuels', 'Renewables & Nuclear']

group_size=[sizes[0]+sizes[1]+sizes[2],sizes[3]+sizes[4]+sizes[5]+sizes[6]+sizes[7]]

subgroup_names = ['Coal','Gas','Oil','Hydro','Nuclear','Wind','Solar','Other Renewables']



for i in range(len(group_names)):

    group_names[i] = str(group_names[i]) + ' ' + str(round(group_size[i],1)) + '%'



for i in range(len(subgroup_names)):

    subgroup_names[i] = str(subgroup_names[i]) + ' ' + str(round(sizes[i],1)) + '%'



subgroup_size=sizes

 

# Create colors

a, b=[plt.cm.Reds, plt.cm.Greens]

 

# First Ring (outside)

fig, ax = plt.subplots(figsize = (10, 10))



ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6)] )

plt.setp( mypie, width=0.3, edgecolor='white')

 

# Second Ring (Inside)

mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.7, colors=[a(0.5), a(0.4), a(0.3), b(0.6), b(0.5), b(0.4), b(0.3), b(0.2)])

plt.setp( mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)

 

# show it

plt.title('2019 Global Electricity Consumption by Energy Source\n\n',size=20)

plt.show()
drop_val = ['Africa','Asia Pacific','CIS','Europe','Europe (other)','Middle East','North America','Other Asia & Pacific','Other S. & Cent. America'

            ,'South & Central America','World','Other Africa','Other CIS','Other Middle East','Other South & Central America',

            'Central America','Eastern Africa','Middle Africa','Other Caribbean',

            'Other Northern Africa','Other South America','Other Southern Africa','USSR','Western Africa']





for i in drop_val:

    df = df[df['Entity'] != i]

#     dfe = dfe[dfe['Entity'] != i]
drop_val = ['Africa','Asia Pacific','CIS','Europe','Europe (other)','Middle East','North America','Other Asia & Pacific','Other S. & Cent. America'

            ,'South & Central America','World','Other Africa','Other CIS','Other Middle East','Other South & Central America',

            'Central America','Eastern Africa','Middle Africa','Other Caribbean',

            'Other Northern Africa','Other South America','Other Southern Africa','USSR','Western Africa']



for i in drop_val:

    df = df[df['Entity'] != i]







col = ['Solar Installed Capacity (gigawatts)',

       'Wind Installed Capacity (gigawatts)',

       'Geothermal Installed Capacity (megawatts)',

       

       'Solar Energy Generation (terawatt-hours)',

       'Wind Energy Generation (terawatt-hours)',

       'Hydropower Energy Generation (terawatt-hours)',

       'Biofuels Energy Generation (terawatt-hours)',

       'Other renewables Energy Generation (terawatt-hours)',

       

       'Renewables Primary Energy (%)', 

       'Renewables Energy Share (%)',

       'Solar Energy Share (%)', 

       'Wind Energy Share (%)',

       'Hydropower Energy Share (%)',

       

       'Solar Electricity Share (%)',

       'Wind Electricity Share (%)', 

       'Hydropower Electricity Share (%)']



for i in col:

    df[i] = df[i].fillna(0)
df['Total Installed Capacity (gigawatts)'] = df['Solar Installed Capacity (gigawatts)']+df['Wind Installed Capacity (gigawatts)']+df['Geothermal Installed Capacity (megawatts)']/1000

df['Total Renewable Energy Generated (terawatt-hours)'] = df['Solar Energy Generation (terawatt-hours)']+df['Wind Energy Generation (terawatt-hours)']+df['Hydropower Energy Generation (terawatt-hours)']+df['Other renewables Energy Generation (terawatt-hours)'] + df['Nuclear Energy Generation (terawatt-hours)']

df['Other Renewables Electricity Share (%)'] = df['Renewables Energy Share (%)'] - df['Solar Electricity Share (%)'] - df['Wind Electricity Share (%)'] - df['Hydropower Electricity Share (%)']

df['Variable Renewable Energy Electricity Share (%)'] = df['Solar Electricity Share (%)']+df['Wind Electricity Share (%)']



df['Total Electricity (TWh)'] = df['Electricity from coal (TWh)'] + df['Electricity from gas (TWh)'] + df['Electricity from hydro (TWh)'] + df['Electricity from nuclear (TWh)'] + df['Electricity from oil (TWh)'] + df['Electricity from other renewables (TWh)'] + df['Electricity from solar (TWh)'] + df['Electricity from wind (TWh)']

df['Nuclear and Renewable Percentage (%)'] = df['Electricity from nuclear (TWh)']/df['Total Electricity (TWh)'] + df['Electricity from hydro (TWh)']/df['Total Electricity (TWh)'] + df['Electricity from other renewables (TWh)']/df['Total Electricity (TWh)'] + df['Electricity from solar (TWh)']/df['Total Electricity (TWh)'] + df['Electricity from wind (TWh)']/df['Total Electricity (TWh)']

df['Nuclear and Renewable Electricity (TWh)'] = df['Electricity from nuclear (TWh)'] + df['Electricity from hydro (TWh)'] + df['Electricity from other renewables (TWh)'] + df['Electricity from solar (TWh)'] + df['Electricity from wind (TWh)']
ext_col = ['Solar Energy Generation (terawatt-hours)',

           'Wind Energy Generation (terawatt-hours)',

           'Hydropower Energy Generation (terawatt-hours)',

           'Biofuels Energy Generation (terawatt-hours)',

           'Other renewables Energy Generation (terawatt-hours)',

           'Total Renewable Energy Generated (terawatt-hours)',

           'Nuclear Energy Generation (terawatt-hours)']



new_col = ['Solar Energy Generation per Capita (megawatt-hours/person)',

           'Wind Energy Generation per Capita (megawatt-hours/person)',

           'Hydropower Energy Generation per Capita (megawatt-hours/person)',

           'Biofuels Energy Generation per Capita (megawatt-hours/person)',

           'Other renewables Energy Generation per Capita (megawatt-hours/person)',

           'Total Renewable Energy Generation per Capita (megawatt-hours/person)',

           'Nuclear Energy Generated per Capita (megawatt-hours/person)',

       

           'Solar Energy Generation / GDP (terawatt-hours/GDP)',

           'Wind Energy Generation / GDP (terawatt-hours/GDP)',

           'Hydropower Energy Generation / GDP (terawatt-hours/GDP)',

           'Biofuels Energy Generation / GDP (terawatt-hours/GDP)',

           'Other renewables Energy Generation / GDP (terawatt-hours/GDP)',

           'Total Renewable Energy Generation / GDP (terawatt-hours/GDP)',

           'Nuclear Energy Generated / GDP (terawatt-hours/GDP)']
# per capita

for i in range(len(ext_col)):

#     print(i)

#     print('New Column:      '+str(new_col[i]))

#     print('Existing Column: '+str(ext_col[i]))

    df[new_col[i]] = df[ext_col[i]]/df['Population']*1e6



# gdp per capita

for i in range(len(ext_col)):

#     print(i)

#     print('New Column:      '+str(new_col[i+7]))

#     print('Existing Column: '+str(ext_col[i]))

    df[new_col[i+7]] = df[ext_col[i]]/df['GDP per capita (2011 USD)']
df_2019 = df[df['Year'] == 2019]

df_2019 = df_2019.reset_index()

df_2017 = df[df['Year'] == 2017]

df_2017 = df_2017.reset_index()



# dfe_2019 = dfe[dfe['Year'] == 2019]

# dfe_2019 = dfe_2019.reset_index()







df_sol_cap = 0

df_wnd_cap = 0

df_geo_cap = 0

df_tot_cap = 0



df_sol_en = 0

df_wnd_en = 0

df_hyd_en = 0

df_bio_en = 0

df_oth_en = 0

df_tot_en = 0



df_sol_en_pc = 0

df_wnd_en_pc = 0

df_hyd_en_pc = 0

df_bio_en_pc = 0

df_oth_en_pc = 0

df_tot_en_pc = 0



df_sol_en_gdp = 0

df_wnd_en_gdp = 0

df_hyd_en_gdp = 0

df_bio_en_gdp = 0

df_oth_en_gdp = 0

df_tot_en_gdp = 0



df_re_pe = 0

df_re_es = 0

df_sol_es = 0

df_wnd_es = 0

df_hyd_es = 0



df_sol_els = 0

df_wnd_els = 0

df_hyd_els = 0

df_oth_els = 0

df_var_els = 0



dataframes = [df_sol_cap,df_wnd_cap,df_geo_cap,df_tot_cap,

df_sol_en,

df_wnd_en,

df_hyd_en,

df_bio_en,

df_oth_en,

df_tot_en,

df_sol_en_pc,

df_wnd_en_pc,

df_hyd_en_pc,

df_bio_en_pc,

df_oth_en_pc,

df_tot_en_pc,



df_re_pe,

df_re_es,

df_sol_es,

df_wnd_es,

df_hyd_es,

df_sol_els,

df_wnd_els,

df_hyd_els,

df_oth_els,

df_var_els]



dataframes_gdp = [df_sol_en_gdp,

df_wnd_en_gdp,

df_hyd_en_gdp,

df_bio_en_gdp,

df_oth_en_gdp,

df_tot_en_gdp]



col = ['Solar Installed Capacity (gigawatts)',

       'Wind Installed Capacity (gigawatts)',

       'Geothermal Installed Capacity (megawatts)',

       'Total Installed Capacity (gigawatts)',

       

       'Solar Energy Generation (terawatt-hours)',

       'Wind Energy Generation (terawatt-hours)',

       'Hydropower Energy Generation (terawatt-hours)',

       'Biofuels Energy Generation (terawatt-hours)',

       'Other renewables Energy Generation (terawatt-hours)',

       'Total Renewable Energy Generated (terawatt-hours)',

       'Nuclear Energy Generation (terawatt-hours)',

       

       'Solar Energy Generation per Capita (megawatt-hours/person)',

        'Wind Energy Generation per Capita (megawatt-hours/person)',

        'Hydropower Energy Generation per Capita (megawatt-hours/person)',

        'Biofuels Energy Generation per Capita (megawatt-hours/person)',

        'Other renewables Energy Generation per Capita (megawatt-hours/person)',

        'Total Renewable Energy Generation per Capita (megawatt-hours/person)',

        'Nuclear Energy Generated per Capita (megawatt-hours/person)',

       

       'Renewables Primary Energy (%)', 

       'Renewables Energy Share (%)',

       'Solar Energy Share (%)', 

       'Wind Energy Share (%)',

       'Hydropower Energy Share (%)',

       

       'Solar Electricity Share (%)',

       'Wind Electricity Share (%)', 

       'Hydropower Electricity Share (%)',

      'Other Renewables Electricity Share (%)',

      'Variable Renewable Energy Electricity Share (%)',

      'Nuclear Electricity Share (%)']



col_gdp = ['Solar Energy Generation / GDP (terawatt-hours/GDP)',

           'Wind Energy Generation / GDP (terawatt-hours/GDP)',

           'Hydropower Energy Generation / GDP (terawatt-hours/GDP)',

           'Biofuels Energy Generation / GDP (terawatt-hours/GDP)',

           'Other renewables Energy Generation / GDP (terawatt-hours/GDP)',

           'Total Renewable Energy Generation / GDP (terawatt-hours/GDP)',

           'Nuclear Energy Generated / GDP (terawatt-hours/GDP)']



df_2019.loc[46,'Renewables Energy Share (%)'] = 100.00



for i in range(len(dataframes)):

    place = df_2019.sort_values(col[i], ascending=False)

    dataframes[i] = place[:10]

#     dataframes[i].set_index('Entity', inplace=True)

#     x2 = dataframes[i]

#     print(x2[col[i]].head(10))

#     print()



for i in range(len(dataframes_gdp)):

    place = df_2017.sort_values(col_gdp[i], ascending=False)

    dataframes_gdp[i] = place[:10]

#     dataframes_gdp[i].set_index('Entity', inplace=True)

#     x2 = dataframes_gdp[i]

#     print(x2[col_gdp[i]].head(10))

#     print()
# fig = plt.figure(figsize = (20, 10)) 

# plt.scatter(x=df_2017['GDP per capita (2011 USD)'],y=df_2019['Nuclear and Renewable Percentage (%)']*100, s=df_2019['Total Electricity (TWh)'],alpha = 0.3)

# plt.show()
# for i in range(2):

#     fig = plt.figure(figsize = (20, 3)) 

#     x = dataframes[i]

#     plt.bar(x['Entity'],x[col[i]])

#     plt.title(col[i])

#     plt.xticks(rotation=45)
# for i in range(4,10):

#     fig = plt.figure(figsize = (20, 3)) 

#     x = dataframes[i]

#     plt.bar(x['Entity'],x[col[i]])

#     plt.title(col[i])

#     plt.xticks(rotation=45)
# for i in range(10,16):

#     fig = plt.figure(figsize = (20, 3)) 

#     x = dataframes[i]

#     plt.bar(x['Entity'],x[col[i]])

#     plt.title(col[i])

#     plt.xticks(rotation=45)
# for i in range(16,24):

#     fig = plt.figure(figsize = (20, 3)) 

#     x = dataframes[i]

#     plt.bar(x['Entity'],x[col[i]])

#     plt.title(col[i])

#     plt.xticks(rotation=45)
fig = plt.figure(figsize = (15, 10))



x = dataframes[9]

x = x[::-1]

t5 = x['Solar Energy Generation (terawatt-hours)']

t6 = x['Solar Energy Generation (terawatt-hours)'] + x['Wind Energy Generation (terawatt-hours)']

t7 = x['Solar Energy Generation (terawatt-hours)'] + x['Wind Energy Generation (terawatt-hours)'] +x['Hydropower Energy Generation (terawatt-hours)']

t8 = x['Solar Energy Generation (terawatt-hours)'] + x['Wind Energy Generation (terawatt-hours)'] +x['Hydropower Energy Generation (terawatt-hours)']+ x['Nuclear Energy Generation (terawatt-hours)']

 

plt.barh(x['Entity'],x['Solar Energy Generation (terawatt-hours)'],color='#f6ab53')

plt.barh(x['Entity'],x['Wind Energy Generation (terawatt-hours)'],left=t5,color='#69b64f')

plt.barh(x['Entity'],x['Hydropower Energy Generation (terawatt-hours)'],left=t6,color='#30b4c9')

plt.barh(x['Entity'],x['Nuclear Energy Generation (terawatt-hours)'],left=t7,color='#f25c5c')

plt.barh(x['Entity'],x['Other renewables Energy Generation (terawatt-hours)'],left=t8,color='#ad85d2')

plt.legend(('Solar','Wind','Hydropower','Nuclear','Other'),prop={'size': 12})



plt.title('Top 10 Renewable Energy Producing Countries\nEnergy Source Breakdown',fontsize=20)

plt.xlabel('Electricity Generation (TWh)', size=15)

plt.xticks(size=13)

plt.yticks(size=13)

plt.show()
fig = plt.figure(figsize = (15, 10)) 



x = dataframes[16]

x = x[::-1]

t5 = x['Solar Energy Generation per Capita (megawatt-hours/person)']

t6 = x['Solar Energy Generation per Capita (megawatt-hours/person)'] + x['Wind Energy Generation per Capita (megawatt-hours/person)']

t7 = x['Solar Energy Generation per Capita (megawatt-hours/person)'] + x['Wind Energy Generation per Capita (megawatt-hours/person)'] +x['Hydropower Energy Generation per Capita (megawatt-hours/person)']

t8 = x['Solar Energy Generation per Capita (megawatt-hours/person)'] + x['Wind Energy Generation per Capita (megawatt-hours/person)'] +x['Hydropower Energy Generation per Capita (megawatt-hours/person)']+ x['Nuclear Energy Generated per Capita (megawatt-hours/person)']

 

plt.barh(x['Entity'],x['Solar Energy Generation per Capita (megawatt-hours/person)'],color='#f6ab53')

plt.barh(x['Entity'],x['Wind Energy Generation per Capita (megawatt-hours/person)'],left=t5,color='#69b64f')

plt.barh(x['Entity'],x['Hydropower Energy Generation per Capita (megawatt-hours/person)'],left=t6,color='#30b4c9')

plt.barh(x['Entity'],x['Nuclear Energy Generated per Capita (megawatt-hours/person)'],left=t7,color='#f25c5c')

plt.barh(x['Entity'],x['Other renewables Energy Generation per Capita (megawatt-hours/person)'],left=t8,color='#ad85d2')

plt.legend(('Solar','Wind','Hydropower','Nuclear','Other'),prop={'size': 12})



plt.title('Top 10 Renewable Energy Producing Countries per Capita\nEnergy Source Breakdown',fontsize=20)

plt.xlabel('Electricity Generation per Capita (MWh/Person)', size=15)

plt.yticks(size=13)

plt.xticks(size=13)

plt.show()
fig = plt.figure(figsize = (15, 10)) 



x = dataframes_gdp[5]

x = x[::-1]

t5 = x['Solar Energy Generation / GDP (terawatt-hours/GDP)']

t6 = x['Solar Energy Generation / GDP (terawatt-hours/GDP)'] + x['Wind Energy Generation / GDP (terawatt-hours/GDP)']

t7 = x['Solar Energy Generation / GDP (terawatt-hours/GDP)'] + x['Wind Energy Generation / GDP (terawatt-hours/GDP)'] +x['Hydropower Energy Generation / GDP (terawatt-hours/GDP)']

t8 = x['Solar Energy Generation / GDP (terawatt-hours/GDP)'] + x['Wind Energy Generation / GDP (terawatt-hours/GDP)'] +x['Hydropower Energy Generation / GDP (terawatt-hours/GDP)']+ x['Nuclear Energy Generated / GDP (terawatt-hours/GDP)']

 

plt.barh(x['Entity'],x['Solar Energy Generation / GDP (terawatt-hours/GDP)'],color='#f6ab53')

plt.barh(x['Entity'],x['Wind Energy Generation / GDP (terawatt-hours/GDP)'],left=t5,color='#69b64f')

plt.barh(x['Entity'],x['Hydropower Energy Generation / GDP (terawatt-hours/GDP)'],left=t6,color='#30b4c9')

plt.barh(x['Entity'],x['Nuclear Energy Generated / GDP (terawatt-hours/GDP)'],left=t7,color='#f25c5c')

plt.barh(x['Entity'],x['Other renewables Energy Generation / GDP (terawatt-hours/GDP)'],left=t8,color='#ad85d2')

plt.legend(('Solar','Wind','Hydropower','Nuclear','Other'),prop={'size': 12})



plt.title('Top 10 Renewable Energy Producing Countries per Capita GDP\nEnergy Breakdown',fontsize=20)

plt.xlabel('Electricity Generation per Capita GDP (TWh/$)', size=15)

plt.yticks(size=13)

plt.xticks(size=13)

plt.show()
fig = plt.figure(figsize = (15, 10)) 





x = df_2019[df_2019['Nuclear and Renewable Percentage (%)']*100 > 90].sort_values('Nuclear and Renewable Percentage (%)',ascending=True)



solar = x['Electricity from solar (TWh)']/x['Total Electricity (TWh)']*100

wind = x['Electricity from wind (TWh)']/x['Total Electricity (TWh)']*100

hydro = x['Electricity from hydro (TWh)']/x['Total Electricity (TWh)']*100

nuclear = x['Electricity from nuclear (TWh)']/x['Total Electricity (TWh)']*100

other = x['Electricity from other renewables (TWh)']/x['Total Electricity (TWh)']*100

 

t5 = solar

t6 = solar + wind

t7 = solar + wind + hydro

t8 = solar + wind + hydro + nuclear



plt.barh(x['Entity'],solar,color='#f6ab53')

plt.barh(x['Entity'],wind,left=t5,color='#69b64f')

plt.barh(x['Entity'],hydro,left=t6,color='#30b4c9')

plt.barh(x['Entity'],nuclear,left=t7,color='#f25c5c')

plt.barh(x['Entity'],other,left=t8,color='#ad85d2')



plt.legend(('Solar','Wind','Hydropower','Nuclear','Other'),loc=(0.5,0.8),prop={'size': 12})



plt.title('Countries with >90% Electricity from Nuclear and Renewable Energy\nEnergy Source Breakdown',fontsize=20)

plt.xlabel('Percentage (%)',size=15)

plt.xlim(0,100)

plt.xticks(size=13)

plt.yticks(size=13)

plt.show()
filt = df_2019['Nuclear and Renewable Percentage (%)'] > .90



print(df_2019.loc[filt,['Entity','Nuclear and Renewable Percentage (%)']].sort_values('Nuclear and Renewable Percentage (%)',ascending=False))