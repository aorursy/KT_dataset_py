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
from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Hide//unhide Code"></form>''')
import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline
worldBank_df_path="/kaggle/input/WDIData.csv"

worldBank_df=pd.read_csv(worldBank_df_path) 
worldBank_df[worldBank_df['Country Name']=='Egypt, Arab Rep.'].loc[150256:150258, 'Indicator Name':'2018']
country_grp=worldBank_df.groupby('Country Name')

Ethiopia_Electrification=country_grp.get_group('Ethiopia').iloc[1,]['1960':'2019']

Egypt_Electrification=country_grp.get_group('Egypt, Arab Rep.').iloc[1,]['1960':'2019']

Sudan_Electrification=country_grp.get_group('Sudan').iloc[1,]['1960':'2019']

Ethiopia_Electrification['2007']=(Ethiopia_Electrification['2006']+Ethiopia_Electrification['2008'])/2       

#Water KPI

Ethiopia_water=worldBank_df[(worldBank_df['Indicator Name']==

                            'People using at least basic drinking water services (% of population)') &

            (worldBank_df['Country Name']=='Ethiopia')]

Egypt_water=worldBank_df[(worldBank_df['Indicator Name']==

                        'People using at least basic drinking water services (% of population)') &

            (worldBank_df['Country Name']=='Egypt, Arab Rep.')]

Sudan_water=worldBank_df[(worldBank_df['Indicator Name']==

                        'People using at least basic drinking water services (% of population)') &

            (worldBank_df['Country Name']=='Sudan')]

Egypt_water=country_grp.get_group('Egypt, Arab Rep.').loc[151212,]['1960':'2019']

Sudan_water=country_grp.get_group('Sudan').loc[332949,]['1960':'2019']

Ethiopia_water=country_grp.get_group('Ethiopia').loc[159798,]['1960':'2019']

#Hydro Power KPI

Ethiopia_hydroPower=worldBank_df[(worldBank_df['Indicator Name']==

                            'Electricity production from hydroelectric sources (% of total)') &

            (worldBank_df['Country Name']=='Ethiopia')]

Egypt_hydroPower=worldBank_df[(worldBank_df['Indicator Name']==

                        'Electricity production from hydroelectric sources (% of total)') &

            (worldBank_df['Country Name']=='Egypt, Arab Rep.')]

Sudan_hydroPower=worldBank_df[(worldBank_df['Indicator Name']==

                        'Electricity production from hydroelectric sources (% of total)') &

            (worldBank_df['Country Name']=='Sudan')]

Egypt_hydroPower=country_grp.get_group('Egypt, Arab Rep.').loc[150605,]['1960':'2019']

Sudan_hydroPower=country_grp.get_group('Sudan').loc[332342,]['1960':'2019']

Ethiopia_hydroPower=country_grp.get_group('Ethiopia').loc[159191,]['1960':'2019']

#Life KPI



Ethiopia_life=worldBank_df[(worldBank_df['Indicator Name']==

                            'Life expectancy at birth, total (years)') &

            (worldBank_df['Country Name']=='Ethiopia')]

Egypt_life=worldBank_df[(worldBank_df['Indicator Name']==

                        'Life expectancy at birth, total (years)') &

            (worldBank_df['Country Name']=='Egypt, Arab Rep.')]

Sudan_life=worldBank_df[(worldBank_df['Indicator Name']==

                        'Life expectancy at birth, total (years)') &

            (worldBank_df['Country Name']=='Sudan')]

Egypt_life=country_grp.get_group('Egypt, Arab Rep.').loc[150953,]['1960':'2019']

Sudan_life=country_grp.get_group('Sudan').loc[332690,]['1960':'2019']

Ethiopia_life=country_grp.get_group('Ethiopia').loc[159539,]['1960':'2019']

#Health KPI

Ethiopia_Health=worldBank_df[(worldBank_df['Indicator Name']==

                           'Current health expenditure per capita (current US$)') &

            (worldBank_df['Country Name']=='Ethiopia')]

Egypt_Health=worldBank_df[(worldBank_df['Indicator Name']==

                        'Current health expenditure per capita (current US$)') &

            (worldBank_df['Country Name']=='Egypt, Arab Rep.')]

Sudan_Health=worldBank_df[(worldBank_df['Indicator Name']==

                        'Current health expenditure per capita (current US$)') &

            (worldBank_df['Country Name']=='Sudan')]

Egypt_Health=country_grp.get_group('Egypt, Arab Rep.').loc[150545,]['1960':'2019']

Sudan_Health=country_grp.get_group('Sudan').loc[332282,]['1960':'2019']

Ethiopia_Health=country_grp.get_group('Ethiopia').loc[159131,]['1960':'2019']

#GDP KPI

Ethiopia_GDP=worldBank_df[(worldBank_df['Indicator Name']==

                            'GDP (current LCU)') &

            (worldBank_df['Country Name']=='Ethiopia')]

Egypt_GDP=worldBank_df[(worldBank_df['Indicator Name']==

                        'GDP (current LCU)') &

            (worldBank_df['Country Name']=='Egypt, Arab Rep.')]

Sudan_GDP=worldBank_df[(worldBank_df['Indicator Name']==

                        'GDP (current LCU)') &

            (worldBank_df['Country Name']=='Sudan')]

Egypt_GDP=country_grp.get_group('Egypt, Arab Rep.').loc[150721,]['1960':'2019']

Sudan_GDP=country_grp.get_group('Sudan').loc[332458,]['1960':'2019']

Ethiopia_GDP=country_grp.get_group('Ethiopia').loc[159307,]['1960':'2019']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,5)) 

#life

axes[1].plot( Egypt_Electrification, linewidth=2, color='limegreen', label='Egypt\nPops=104 mil')

axes[1].plot( Sudan_Electrification, linewidth=2, color='orange', label='Sudan\nPops=45 mil')

axes[1].plot( Ethiopia_Electrification, linewidth=2, color='red', label='Ethiopia\nPops=108 mil')

axes[1].set_title("Access to electricity (% of population)",fontweight='bold', fontsize=14)

axes[1].set_facecolor('#EAEAF2')

axes[1].spines["right"].set_visible(False)    

axes[1].spines["left"].set_visible(False) 

axes[1].spines["top"].set_visible(False)    

axes[1].spines["bottom"].set_visible(False)  

axes[1].grid(axis='both',color='w', linestyle='--') 

axes[1].set_ylabel('%', fontweight='bold', size=13)

axes[1].set_xlabel('year', fontweight='bold', size=12)

axes[1].set_xticks(Ethiopia_Electrification.index[30::3]) 

axes[1].set_yticks(range(0,105,10))

axes[1].xaxis.set_tick_params(labelsize=12)

legend_properties = {'weight':'bold', 'size': 14}

axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),

               fancybox=True, framealpha=0.0, ncol=3, prop=legend_properties)

plt.rcParams["axes.labelweight"] = "bold"







#share

group_names=['Egypt\'s share (66%)', 'Sudan\'s share (22%)', 'Evap(12%)']

group_size=[66,22,12]

subgroup_names=["","",""]

subgroup_size=[66,22,12]

explode = (0.1, 0.0, 0.1)

 

# Create colors

a, b, c=[plt.cm.Blues, plt.cm.Blues, plt.cm.Oranges_r] 

 

# First Ring (outside)

#fig, ax = plt.subplots()

#ax.axis('equal')

mypie, _ = axes[0].pie(group_size, radius=1.0, labels=group_names, colors=['#3964db', '#ffd4a3', '#EAEAF2'],

                 textprops={'color':"k", 'fontsize':12},

                       explode=explode, startangle=0 )

plt.setp( mypie, width=0.3, edgecolor='white')



# Second Ring (Inside)

mypie2, _ = axes[0].pie(subgroup_size, radius=1.3-0.6, labels=subgroup_names, 

                   labeldistance=0.0, colors=['#3964db', '#ffd4a3', '#EAEAF2'], 

                   textprops={'color':"k",'fontweight':'bold'}, 

                        explode=explode, startangle=0)  

plt.setp( mypie2, width=0.4, edgecolor='white')

axes[0].set_title("Share of Abay (aka Blue-Nile):\nEthiopia = 0%, \nSudan(22%), \nEgypt(66%), \nEvaporation(12%)",

                  fontweight='bold', fontsize=14)



#axes[0].annotate('s','ss')





#fig.set_size_inches(12,12)

 

# show it

plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

#life

axes[0,0].plot( Egypt_life, linewidth=3, color='limegreen', label='Egypt')

axes[0,0].plot( Sudan_life, linewidth=3, color='orange', label='Sudan')

axes[0,0].plot( Ethiopia_life, linewidth=3, color='red', label='Ethiopia')

axes[0,0].set_title("Life expectancy at birth, total (years)",fontweight='bold', fontsize=14)

axes[0,0].set_facecolor('#EAEAF2')

axes[0,0].spines["right"].set_visible(False)    

axes[0,0].spines["left"].set_visible(False) 

axes[0,0].spines["top"].set_visible(False)    

axes[0,0].spines["bottom"].set_visible(False)  

axes[0,0].grid(axis='both',color='w', linestyle='--') 

axes[0,0].set_ylabel('years', fontweight='bold', size=13)

axes[0,0].set_xticks(Sudan_life.index[0::5]) 

axes[0,0].set_yticks(range(0,101,10))

axes[0,0].xaxis.set_tick_params(labelsize=12)

plt.rcParams["axes.labelweight"] = "bold"

#Health

axes[0,1].plot( Egypt_Health, linewidth=3, color='limegreen', label='Egypt')

axes[0,1].plot( Sudan_Health, linewidth=3, color='orange', label='Sudan')

axes[0,1].plot( Ethiopia_Health, linewidth=3, color='red', label='Ethiopia')

axes[0,1].set_title("Current health expenditure per capita (current US$)",fontweight='bold', fontsize=14)

axes[0,1].set_facecolor('#EAEAF2')

axes[0,1].spines["right"].set_visible(False)    

axes[0,1].spines["left"].set_visible(False) 

axes[0,1].spines["top"].set_visible(False)    

axes[0,1].spines["bottom"].set_visible(False)  

axes[0,1].grid(axis='both',color='w', linestyle='--') 

axes[0,1].set_ylabel('$', fontweight='bold', size=13)

axes[0,1].set_xticks(Sudan_life.index[40::2]) 

#axes[0,1].set_yticks(range(0,190,10))

axes[0,1].xaxis.set_tick_params(labelsize=12)

plt.rcParams["axes.labelweight"] = "bold"



#water

axes[1,0].plot( Egypt_water, linewidth=3, color='limegreen', label='Egypt')

axes[1,0].plot( Sudan_water, linewidth=3, color='orange', label='Sudan')

axes[1,0].plot( Ethiopia_water, linewidth=3, color='red', label='Ethiopia')

axes[1,0].set_title("People using at least basic drinking water services (% of population)",fontweight='bold', fontsize=14)

axes[1,0].set_facecolor('#EAEAF2')

axes[1,0].spines["right"].set_visible(False)    

axes[1,0].spines["left"].set_visible(False) 

axes[1,0].spines["top"].set_visible(False)    

axes[1,0].spines["bottom"].set_visible(False)  

axes[1,0].grid(axis='both',color='w', linestyle='--') 

axes[1,0].set_ylabel('%', fontweight='bold', size=13)

axes[1,0].set_xticks(Sudan_Electrification.index[40::2]) 

#axes[0,1].set_yticks(range(0,190,10))

axes[1,0].xaxis.set_tick_params(labelsize=12)

plt.rcParams["axes.labelweight"] = "bold"

# water People using at least basic drinking water services (% of population)

axes[1,1].plot( Egypt_hydroPower, linewidth=3, color='limegreen', label='Egypt')

axes[1,1].plot( Sudan_hydroPower, linewidth=3, color='orange', label='Sudan')

axes[1,1].plot( Ethiopia_hydroPower, linewidth=3, color='red', label='Ethiopia')

axes[1,1].set_title("'Electricity production from hydroelectric sources (% of total)",fontweight='bold', fontsize=14)

axes[1,1].set_facecolor('#EAEAF2')

axes[1,1].spines["right"].set_visible(False)    

axes[1,1].spines["left"].set_visible(False) 

axes[1,1].spines["top"].set_visible(False)    

axes[1,1].spines["bottom"].set_visible(False)  

axes[1,1].grid(axis='both',color='w', linestyle='--') 

axes[1,1].set_ylabel('%', fontweight='bold', size=13)

axes[1,1].set_xticks(Sudan_hydroPower.index[12::4]) 

#axes[0,1].set_yticks(range(0,190,10))

axes[1,1].xaxis.set_tick_params(labelsize=12)

plt.rcParams["axes.labelweight"] = "bold"



plt.show()