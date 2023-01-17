import requests

import lxml.html as lh

import pandas as pd

import re
#%load pd_option.py

import pandas as pd



def start():

    options = {

        'display': {

            'max_columns': None,

            'max_colwidth': 20,

            'max_rows': 10,

            'precision': 4,

            'show_dimensions': False

        },

        'mode': {

            'chained_assignment': None   # Controls SettingWithCopyWarning

        }

    }



    for category, option in options.items():

        for op, value in option.items():

            pd.set_option(f'{category}.{op}', value)  # Python 3.6+



if __name__ == '__main__':

    start()

    del start  # Clean up namespace in the interpreter

    

pd.set_option('display.float_format', '{:.0f}'.format)
url3="https://www.worldometers.info/coronavirus/"

page3=requests.get(url3)

doc3 = lh.fromstring(page3.content)

world = doc3.xpath('//*//tr')

#tr_elements = doc.xpath('//*[@id="thetable"]//tbody//tr')

#topdata=doc2.xpath('//li[@class="FeaturedStats_stat__1MPv_"]//text()')
pol3=[]

i=0

#For each row, store each first element (header) and an empty list

for x in world[0]:

    i+=1

    y=x.text_content()

    y=re.sub('[\n]', '', y)

    #print ('%d:"%s"'%(i,y))

    pol3.append((y,[]))
for j in range(9,219):

    i=0

    for x in world[j]:

        y=x.text_content()

        y=re.sub('[\n]', '', y) # Replace newline

        y=re.sub(',', '', y) # Remove thousand seperators

        y=y.strip() # Remove leading and trailing zeros

        y=re.sub(' ', '', y) # Remove empty spaces

        #print ('%d:"%s"'%(i,y))

        pol3[i][1].append((y))

        i+=1
Dict3={title:column for (title,column) in pol3}

df3=pd.DataFrame(Dict3)

df3.head(3)
df=df3.drop('#', axis=1)

df=df.rename(columns={'Country,Other' : 'Country'})

df.set_index("Country", drop=True, inplace=True)

df.shape
# Find the country with no population, and drop them

countryWithNoPopulation=df[df['Population'].map(len)<1].index 

df.drop(countryWithNoPopulation, inplace=True)

# Find and drop country with no death and drop them

countryWithNoDeath=df[df['TotalDeaths'].map(len)<1].index 

df.drop(countryWithNoDeath, inplace=True)
df['TotalCases']=df['TotalCases'].astype('int')

df['Population']=df['Population'].astype('int')

df['TotalDeaths']=df['TotalDeaths'].astype('int')
from datetime import date

today=date.today()

todayis=today.strftime("%d/%m/%Y")

df['Date']=todayis

df.to_csv('Covid_data.csv', mode='a', header=False)
df['CasesPerK']=(df['TotalDeaths']/df['Population'])*1000

df['DeathPerK']=(df['TotalDeaths']/df['TotalCases'])*1000
#pd.options.display.float_format = '{:,}'.format

pd.set_option('display.float_format', '{0:,.3f}'.format)
disp=df[['CasesPerK','DeathPerK' ]]
disp['TotalCases']=df['TotalCases'].apply(lambda x: '{0:,}'.format(x))

disp['TotalDeaths']=df['TotalDeaths'].apply(lambda x: '{0:,}'.format(x))

disp['Population']=df['Population'].apply(lambda x: '{0:,}'.format(x))
disp.head()
df=df.reset_index()
dispn=df[['Country','Population', 'TotalCases', 'TotalDeaths', 'CasesPerK','DeathPerK']]
top10=dispn.nlargest(10, ['TotalDeaths','TotalCases'], keep='first')

top10.to_csv("top10.csv")
pd.set_option('display.max_rows', None)

top150=dispn.nlargest(150, ['TotalDeaths','TotalCases'], keep='first').reset_index(drop=True)

#top150
pd.set_option('display.max_rows', None)

top150=dispn.nlargest(150, ['TotalCases','TotalDeaths'], keep='first').reset_index(drop=True)
top10
pd.set_option('display.max_rows', None)

top150=dispn.nlargest(150, ['CasesPerK','TotalDeaths'], keep='first').reset_index(drop=True)

#top150
ax = top10.plot.bar(x='Country', y='TotalCases', rot=0, figsize=(15, 10), colormap='jet')

ax.set_ylabel("Number of Cases")

ax.set_title('Corona Cases - Top 10 Country', fontsize=24 )

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.03))

fig = ax.get_figure()

fig.savefig('coronaCases.png')
top10_cases=top10[['Country','TotalCases', 'TotalDeaths']]
cased=top10_cases.plot.bar(x='Country', rot=0, figsize=(15, 10),colormap='jet')

cased.set_title('CORONA CASE to Death Ratio - TOP 10 COUNTRY', fontsize=24 )
case_rates=top10[['Country', 'CasesPerK']].sort_values('CasesPerK',ascending=False)
cr=case_rates.plot.bar( x='Country', rot=0, figsize=(15, 10))

cr.set_title('CORONA CASE in Every 1000 - TOP 10 COUNTRY', fontsize=24 )
death=top10[['Country', 'TotalDeaths']].sort_values('TotalDeaths',ascending=False)

death.to_csv("US and Top 10 Country with Covid Death.csv")
import matplotlib.pyplot as plt

ax=death.plot.bar( x='Country', rot=0, figsize=(15, 10))

ax.set_title('CORONA Number of Death - TOP 10 COUNTRY', fontsize=24 )

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.01))

fig = ax.get_figure()

fig.savefig('coronaDeath.png')
import numpy as np

top10['Death%']=top10['DeathPerK']/100

deathR=top10[['Country', 'Death%' ]].sort_values('Death%')

dr=deathR.plot.bar( x='Country', rot=0, figsize=(15, 10))

dr.set_title('CORONA Death Rate % of Cases - TOP 10 COUNTRY', fontsize=24 )



for p in dr.patches: 

    dr.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), \

    ha='center', va='center', xytext=(0, 10), textcoords='offset points') 

fig = dr.get_figure()

fig.savefig('coronaDeathRate.png')