# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



matplotlib.style.use('ggplot')

# Get current size

fig_size = plt.rcParams["figure.figsize"]

 

fig_size[0] = 8

fig_size[1] = 6

plt.rcParams["figure.figsize"] = fig_size

print ("Current size:", fig_size)

# Any results you write to the current directory are saved as output.
d = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")

print(d.head())
df = pd.DataFrame(d['Date'])

df = pd.DataFrame(df['Date'].str.split('/').tolist(), columns = ['Month','Day','Year'])

df = pd.DataFrame(df.groupby('Year', as_index=False).size())

df.columns = ['Number of crushes']



df.plot(kind = 'bar', figsize=(10,9))
df = d['Location'].str[0:].str.split(',', expand=True)

d['Country'] = df[3].fillna(df[2]).fillna(df[1]).str.strip()

usNames = ['Virginia','New Jersey','Ohio','Pennsylvania', 'Maryland', 'Indiana', 'Iowa',

          'Illinois','Wyoming', 'Minnisota', 'Wisconsin', 'Nevada', 'NY','California',

          'WY','New York','Oregon', 'Idaho', 'Connecticut','Nebraska', 'Minnesota', 'Kansas',

          'Texas', 'Tennessee', 'West Virginia', 'New Mexico', 'Washington', 'Massachusetts',

          'Utah', 'Ilinois','Florida', 'Michigan', 'Arkansas','Colorado', 'Georgia','Missouri',

          'Montana', 'Mississippi','Alaska','Jersey', 'Cailifornia', 'Oklahoma','North Carolina',

          'Kentucky','Delaware','D.C.','Arazona','Arizona','South Dekota','New Hampshire','Hawaii',

          'Washingon','Massachusett','Washington DC','Tennesee','Deleware','Louisiana',

          'Massachutes', 'Louisana', 'New York (Idlewild)','Oklohoma','North Dakota','Rhode Island',

          'Maine','Alakska','Wisconson','Calilfornia','Virginia','Virginia.','CA','Vermont',

          'HI','AK','IN','GA','Coloado','Airzona','Alabama','Alaksa' 

          ]



afNames = ['Afghanstan'] #Afghanistan

anNames = ['off Angola'] #Angola

ausNames = ['Qld. Australia','Queensland  Australia','Tasmania','off Australia'] #Australia

argNames = ['Aregntina'] #Argentina

azNames = ['Azores (Portugal)'] #Azores

baNames = ['Baangladesh'] #Bangladesh

bahNames = ['Great Inagua'] #Bahamas

berNames = ['off Bermuda'] #Bermuda

bolNames = ['Boliva','BO'] #Bolivia

bhNames = ['Bosnia-Herzegovina'] #Bosnia Herzegovina

bulNames = ['Bugaria','Bulgeria'] #Bulgaria

canNames = ['British Columbia', 'British Columbia Canada','Canada2',

            'Saskatchewan','Yukon Territory'] #Canada

camNames = ['Cameroons','French Cameroons'] #Cameroon

caNames = ['Cape Verde Islands'] #Cape Verde

chNames = ['Chili'] #Chile

coNames = ['Comoro Islands', 'Comoros Islands'] #Comoros

djNames = ['Djbouti','Republiof Djibouti'] #Djibouti

domNames = ['Domincan Republic', 'Dominica'] #Dominican Republic

drcNames = ['Belgian Congo','Belgian Congo (Zaire)','Belgium Congo'

           'DR Congo','DemocratiRepubliCogo','DemocratiRepubliCongo',

            'DemocratiRepubliof Congo','DemoctratiRepubliCongo','Zaire',

           'Zaïre'] #Democratic Republic of Congo

faNames = ['French Equitorial Africa'] #French Equatorial Africa

gerNames = ['East Germany','West Germany'] #Germany

grNames = ['Crete'] #Greece

haNames = ['Hati'] #Haiti

hunNames = ['Hunary'] #Hungary

inNames = ['Indian'] #India

indNames = ['Inodnesia','Netherlands Indies'] #Indonesia

jamNames = ['Jamacia'] #Jamaica

malNames = ['Malaya'] #Malaysia

manNames = ['Manmar'] #Myanmar

marNames = ['Mauretania'] #Mauritania

morNames = ['Morrocco','Morroco'] #Morocco

nedNames = ['Amsterdam','The Netherlands'] #Netherlands

niNames = ['Niger'] #Nigeria

philNames = ['Philipines','Philippine Sea', 'Phillipines',

            'off the Philippine island of Elalat'] #Philippines

romNames = ['Romainia'] #Romania

rusNames = ['Russian','Soviet Union','USSR'] #Russia

saNames = ['Saint Lucia Island'] #Saint Lucia

samNames = ['Western Samoa'] #Samoa

siNames = ['Sierre Leone'] #Sierra Leone

soNames = ['South Africa (Namibia)'] #South Africa

surNames = ['Suriname'] #Surinam

uaeNames = ['United Arab Emirates'] #UAE

ukNames = ['England', 'UK','Wales','110 miles West of Ireland'] #United Kingdom

uvNames = ['US Virgin Islands','Virgin Islands'] #U.S. Virgin Islands

wkNames = ['325 miles east of Wake Island']#Wake Island

yuNames = ['Yugosalvia'] #Yugoslavia

zimNames = ['Rhodesia', 'Rhodesia (Zimbabwe)'] #Zimbabwe



clnames = []

for country in d['Country'].values:

    if country in afNames:

        clnames.append('Afghanistan')

    elif country in anNames:

        clnames.append('Angola')

    elif country in ausNames:

        clnames.append('Australia')

    elif country in argNames:

        clnames.append('Argentina')

    elif country in azNames:

        clnames.append('Azores')

    elif country in baNames:

        clnames.append('Bangladesh')

    elif country in bahNames:

        clnames.append('Bahamas')

    elif country in berNames:

        clnames.append('Bermuda')

    elif country in bolNames:

        clnames.append('Bolivia')

    elif country in bhNames:

        clnames.append('Bosnia Herzegovina')

    elif country in bulNames:

        clnames.append('Bulgaria')

    elif country in canNames:

        clnames.append('Canada')

    elif country in camNames:

        clnames.append('Cameroon')

    elif country in caNames:

        clnames.append('Cape Verde')

    elif country in chNames:

        clnames.append('Chile')

    elif country in coNames:

        clnames.append('Comoros')

    elif country in djNames:

        clnames.append('Djibouti')

    elif country in domNames:

        clnames.append('Dominican Republic')

    elif country in drcNames:

        clnames.append('Democratic Republic of Congo')

    elif country in faNames:

        clnames.append('French Equatorial Africa')

    elif country in gerNames:

        clnames.append('Germany')

    elif country in grNames:

        clnames.append('Greece')

    elif country in haNames:

        clnames.append('Haiti')

    elif country in hunNames:

        clnames.append('Hungary')

    elif country in inNames:

        clnames.append('India')

    elif country in jamNames:

        clnames.append('Jamaica')

    elif country in malNames:

        clnames.append('Malaysia')

    elif country in manNames:

        clnames.append('Myanmar')

    elif country in marNames:

        clnames.append('Mauritania')

    elif country in morNames:

        clnames.append('Morocco')

    elif country in nedNames:

        clnames.append('Netherlands')

    elif country in niNames:

        clnames.append('Nigeria')

    elif country in philNames:

        clnames.append('Philippines')

    elif country in romNames:

        clnames.append('Romania')

    elif country in rusNames:

        clnames.append('Russia')

    elif country in saNames:

        clnames.append('Saint Lucia')

    elif country in samNames:

        clnames.append('Samoa')

    elif country in siNames:

        clnames.append('Sierra Leone')

    elif country in soNames:

        clnames.append('South Africa')

    elif country in surNames:

        clnames.append('Surinam')

    elif country in uaeNames:

        clnames.append('UAE')

    elif country in ukNames:

        clnames.append('United Kingdom')

    elif country in usNames:

        clnames.append('United States of America')

    elif country in uvNames:

        clnames.append('U.S. Virgin Islands')

    elif country in wkNames:

        clnames.append('Wake Island')

    elif country in yuNames:

        clnames.append('Yugoslavia')

    elif country in zimNames:

        clnames.append('Zimbabwe')

    else:

        clnames.append(country)

        

d['Cleaned Country'] = clnames        

fatalcountries = d[['Fatalities','Cleaned Country']].groupby(['Cleaned Country']).agg('sum')

fatalcountries.reset_index(inplace = 'True')

operator = d[['Operator','Fatalities']].groupby('Operator').agg(['sum','count'])

fatalities = operator['Fatalities','sum'].sort_values(ascending=False)

totalfatal = fatalities.sum()

fatalcountries['Proportion of Total'] = fatalcountries['Fatalities']/totalfatal



fig_c, (ax1,ax2) = plt.subplots(2,1,sharex = True)

fatalcountries = fatalcountries.sort_values('Fatalities', ascending=False)

fatalcountries[fatalcountries['Fatalities']>1000].plot(x = 'Cleaned Country'

                                                     , y = 'Fatalities'

                                                     , ax = ax1

                                                     , kind = 'bar'

                                                     , grid = True)

fatalcountries[fatalcountries['Fatalities']>1000].plot(x = 'Cleaned Country'

                                                     , y = 'Proportion of Total'

                                                     , ax = ax2

                                                     , kind = 'bar'

                                                     , grid = True)
df = pd.DataFrame(d['Date'].str.split('/').tolist(), columns = ['Month','Day','Year'])

d1972 = d[df['Year']=='1972'].reset_index(drop=True)

d1972 = pd.DataFrame(d1972.groupby('Operator', as_index=False).size())

d1972.plot(kind = 'bar', figsize=(14,9))
df = pd.DataFrame(d['Time'])

df= df.dropna().reset_index(drop=True)

df = pd.DataFrame(df['Time'].str.split(':',1).tolist(), columns = ['Hour','Minute'])

df = pd.DataFrame(df.groupby('Hour', as_index=False).size())

df.columns = ['Number of crushes']
df = pd.DataFrame(d[['Date']])

df = pd.DataFrame(df['Date'].str.split('/').tolist(), columns = ['Month','Day','Year'])

df = df.join(d['Fatalities'])

print(df)

df = pd.DataFrame(df.groupby('Year', as_index=False)['Fatalities'].sum())

#df.plot(x='Year', y='Fatalities')
df = pd.DataFrame(d['Operator'])

df = pd.DataFrame(df.groupby('Operator', as_index=False).size())

df.columns = ['Number of crushes']

df = df.sort_values('Number of crushes', ascending=False)

df[:20].plot(kind='bar')
df = pd.DataFrame(d[['Operator','Fatalities']])

df = pd.DataFrame(df.groupby('Operator', as_index=False).sum())

df = df.sort_values('Fatalities', ascending=False)

df[:20].plot(x='Operator', kind='bar')
df = pd.DataFrame(d['Type'])

df = pd.DataFrame(df.groupby('Type', as_index=False).size())

df.columns = ['Number of crushes']

df = df.sort_values('Number of crushes', ascending=False)

df[:20].plot(kind='bar')
df = pd.DataFrame(d['Date'][d['Operator']=='Aeroflot'])

df = pd.DataFrame(df['Date'].str.split('/').tolist(), columns = ['Month','Day','Year'])

df = pd.DataFrame(df.groupby('Year', as_index=False).size())

df.columns = ['Number of crushes']



df.plot()
df = pd.DataFrame(d['Date'][d['Operator']=='Aeroflot'])

df = pd.DataFrame(df['Date'].str.split('/').tolist(), columns = ['Month','Day','Year'])

df = df.join(d['Fatalities'][d['Operator']=='Aeroflot'].reset_index(drop=True))

df = df.replace(to_replace='NaN', value=0)

df = pd.DataFrame(df.groupby('Year', as_index=False)['Fatalities'].sum())

df.plot(x='Year', y='Fatalities')
df = pd.DataFrame(d['Date'].str.split('/').tolist(), columns = ['Month','Day','Year'])

dAeroflot = d[df['Year']=='1973'][d['Operator']=='Aeroflot'].reset_index(drop=True)

print(dAeroflot)
df = pd.DataFrame(d[['Type','Operator']])

df = pd.DataFrame(data={'Type':df.groupby(['Type','Operator']).size().index.get_level_values('Type'),'Operator':df.groupby(['Type','Operator']).size().index.get_level_values('Operator'),'Count':df.groupby(['Type','Operator']).size()})

df = df.reset_index(drop=True)

df =df.sort_values('Count', ascending=False)

df = df[df['Count']>5].pivot(index='Type', columns='Operator', values='Count')

df = df.replace(to_replace='NaN', value=0)

plt.pcolor(df, cmap='Reds')

plt.yticks(np.arange(0.5, len(df.index), 1), df.index)

plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation='vertical')

plt.show()
df = pd.DataFrame(d[['Aboard','Fatalities']])

df = df.sort_values('Aboard').dropna().reset_index(drop=True)

df['Alive'] = df['Aboard']-df['Fatalities']

dt = pd.DataFrame(df['Aboard'])

del df['Aboard']

df.loc[:,:] = df.loc[:,:].div(df.sum(axis=1), axis=0)

df = df.dropna().reset_index(drop=True)

df['Fatalities'] = df['Fatalities'].round(2)

df = pd.DataFrame(df.groupby(['Fatalities']).size())

df.columns = ['Number of crushes']

df.plot(kind='area')