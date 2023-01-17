from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','city':'City','attacktype1_txt':'AttackType','nkill':'Killed','nwound':'Wounded','targtype1_txt':'TargetType','success':'Successful'},inplace=True)

data = data[['Year','Month','Day','Country','Region','City','AttackType','Killed','Wounded','TargetType','Successful']]

data['Casualities']=data['Killed']+data['Wounded']

#data.head(10)



civilData = data[(data.TargetType == 'Private Citizens & Property')]



successattacks = data['Successful'].value_counts().tolist()

topcountries = data['Country'].value_counts()[:10].keys().tolist()



print('Total successful terrorist attacks: ',successattacks[0])

print('Total unsuccessful terrorist attacks: ',successattacks[1])

print('Total success rate: ',"{0:.2f}".format((successattacks[0]/(successattacks[0]+successattacks[1]))*100)+'%')



country=data['Country'].value_counts()[:10].to_frame()

country.columns=['Attacks']

successful=data.groupby('Country')['Successful'].sum().to_frame()

country.merge(successful,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.suptitle('Success Rate of Terrorist Attacks in Countries')

plt.show()



country=civilData['Country'].value_counts()[:10].to_frame()

country.columns=['Attacks']

successful=civilData.groupby('Country')['Successful'].sum().to_frame()

country.merge(successful,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.suptitle('Success Rate of Civilian-targeted Terrorist Attacks in Countries')

plt.show()





country=data['Country'].value_counts()[:10].to_frame()

country.columns=['Attacks']

killed=data.groupby('Country')['Killed'].sum().to_frame()

country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.suptitle('Number of attacks and people killed in Countries')

plt.show()



country=civilData['Country'].value_counts()[:10].to_frame()

country.columns=['Attacks']

killed=civilData.groupby('Country')['Killed'].sum().to_frame()

country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.suptitle('Number of Civilian-targeted attacks and Civilians killed in Countries')

plt.show()



countryattacks = data['Country'].value_counts().keys().tolist()

countryattacksnum = data['Country'].value_counts().tolist()

civilcountryattacks = civilData['Country'].value_counts().keys().tolist()

civilcountryattacksnum = civilData['Country'].value_counts().tolist()



successcountry = data.groupby('Country')['Successful'].sum()

civilsuccesscountry = civilData.groupby('Country')['Successful'].sum()

killedcountry = data.groupby('Country')['Killed'].sum()

civilkilledcountry = civilData.groupby('Country')['Killed'].sum()



countrykilled = killedcountry.keys().tolist()

countrykillednum = killedcountry.tolist()



print('Success Rate in the Top Ten Countries')

for x in range(0,10):

    print(countryattacks[x],' : ',"{0:.2f}".format(successcountry[countryattacks[x]]/countryattacksnum[x]*100),'%')



print('\nSuccess Rate (Civilian) in the Top Ten Countries')

for x in range(0,10):

    print(civilcountryattacks[x],' : ',"{0:.2f}".format(civilsuccesscountry[civilcountryattacks[x]]/civilcountryattacksnum[x]*100),'%')    

    

print('\nKilled to Attack Ratio in the Top Ten Countries')

for x in range(0,10):

    print(countryattacks[x],' : ',"{0:.2f}".format(killedcountry[countryattacks[x]]/countryattacksnum[x]))

 

print('\nKilled to Attack Ratio (Civilian) in the Top Ten Countries')

for x in range(0,10):

    print(civilcountryattacks[x],' : ',"{0:.2f}".format(civilkilledcountry[civilcountryattacks[x]]/civilcountryattacksnum[x]))

country=data['Region'].value_counts()[:5].to_frame()

country.columns=['Attacks']

killed=data.groupby('Region')['Successful'].sum().to_frame()

country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.suptitle('Success Rate of Terrorist Attacks in Regions')

plt.show()



country=data['Region'].value_counts()[:5].to_frame()

country.columns=['Attacks']

killed=data.groupby('Region')['Killed'].sum().to_frame()

country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.suptitle('Number of attacks vs Number of killed in Regions')

plt.show()



regionattacks = data['Region'].value_counts().keys().tolist()

regionattacksnum = data['Region'].value_counts().tolist()



successregion = data.groupby('Region')['Successful'].sum()

killedregion = data.groupby('Region')['Killed'].sum()



print('Success Rate in the Top Five Regions')

for x in range(0,5):

    print(regionattacks[x],' : ',"{0:.2f}".format(successregion[regionattacks[x]]/regionattacksnum[x]*100),'%')



print('\nKilled to Attack Ratio in the Top Five Regions')

for x in range(0,5):

    print(regionattacks[x],' : ',"{0:.2f}".format(killedregion[regionattacks[x]]/regionattacksnum[x]))
data = [ dict(

        type='choropleth',

        #autocolorscale = False,

        locations = countryattacks,

        z = countryattacksnum,

        text = countryattacks,

        locationmode = 'country names',

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Attacks")

        ) ]



layout = dict(

        title = 'Terrorism Around The World',

        geo = dict(

            scope='world',

            projection=dict( type='Mercator' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )



data = [ dict(

        type='choropleth',

        #autocolorscale = False,

        locations = countrykilled,

        z = countrykillednum,

        text = countrykilled,

        locationmode = 'country names',

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Killed")

        ) ]



layout = dict(

        title = 'People killed by Terrorism Around The World',

        geo = dict(

            scope='world',

            projection=dict( type='Mercator' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )