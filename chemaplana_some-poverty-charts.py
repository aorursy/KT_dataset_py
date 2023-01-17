# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pov_file = pd.read_csv('../input/data.csv')

pov_file.columns
pov_file.info()
pov_file.describe()

pov_file.head()
pov = pov_file

columns = ['Country', 'CCode', 'IName',

       'ICode', '1974', '1975', '1976', '1977', '1978', '1979',

       '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',

       '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',

       '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003',

       '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011',

       '2012', '2013', '2014', '2015', 'Blank']

pov.columns = columns

print ('ICodes', len(pov['ICode'].unique()))

print ('CCodes', len(pov['CCode'].unique()))

print ('Countries', len(pov['Country'].unique()))
print (pov.isnull().sum())

print (pov.isnull().sum().sum())

pov = pov.drop(['Blank'], axis=1)

pov['Counter'] = pov.apply(lambda x: x.count(), axis=1)

pov = pov[pov['Counter'] > 4]

print (pov.isnull().sum())

print (pov.isnull().sum().sum())
pov_group = pov.loc[:,['Country', 'CCode', 'IName', 'ICode', 'Counter']]

pov_group = pov_group.groupby('Country').sum()

pov_group = pov_group.sort_values(by='Counter', ascending=False)

pov_group
print (pov['Country'].unique())
import matplotlib.pyplot as plt

import matplotlib.patches as patches

plt.style.use('ggplot')



pov_world = pov[pov['Country'] == 'World']

print (pov_world['ICode'].unique())

fig, ax = plt.subplots(figsize=(18,8))

plt.subplots_adjust(left=0.2, right=0.8)

for year in range(1974, 2015):

	ax.scatter(year, pov_world.loc[pov_world['ICode']=='SP.POP.TOTL', str(year)], c='Blue')

	ax.scatter(year, pov_world.loc[pov_world['ICode']=='SI.POV.NOP1', str(year)]*1000000, c='Red')

	ax.set_title('World')

	ax.grid(c='Grey', alpha=1)

recs=[]

class_colour=['b','r']

for i in range(2):

	recs.append(patches.Rectangle((0,0),1,1,fc=class_colour[i]))

fig.legend(recs, ('Total Population', 'Pop under $1.9/d'), loc=(0.15,0.8), 

           fontsize=10)

country_list = ['East Asia & Pacific','Europe & Central Asia', 'Latin America & Caribbean', 

 'Middle East & North Africa', 'South Asia', 'Sub-Saharan Africa']



pov_country_list = pov[pov['Country'].isin(country_list)]

for coun in country_list:

	pov_country = pov_country_list[pov_country_list['Country']==coun]

	print (coun, pov_country['ICode'].unique())



rows= 3

cols = 2



fig, ax = plt.subplots(rows, cols, figsize=(20,10))

plt.subplots_adjust(left=0.2, right=0.8, hspace=0.5)

for r in range(rows):

	for c in range(cols):

		pov_country = pov_country_list[pov_country_list['Country']==country_list[cols*r+c]]

		for year in range(1974, 2015):

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SP.POP.TOTL', str(year)],

                             c='b')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP1', str(year)]*1000000,

                             c='r')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP2', str(year)]*1000000,

                             c='g')

			ax[r, c].set_title(country_list[cols*r+c])

			ax[r, c].grid(c='Grey', alpha=1)

recs=[]

class_colour=['b','r','g']

for i in range(3):

	recs.append(patches.Rectangle((0,0),1,1,fc=class_colour[i]))

fig.legend(recs, ('Total Population', 'Pop under $1.9/d', 'Pop under $3.1/d'), loc=(0.815,0.92),

           fontsize=10)

rows= 3

cols = 2



fig, ax = plt.subplots(rows, cols, figsize=(20,10))

plt.subplots_adjust(left=0.2, right=0.8, hspace=0.5)

for r in range(rows):

	for c in range(cols):

		pov_country = pov_country_list[pov_country_list['Country']==country_list[cols*r+c]]

		for year in range(1974, 2015):

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SP.POP.TOTL', str(year)],

                             c='b')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP1', str(year)]*1000000,

                             c='r')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP2', str(year)]*1000000,

                             c='g')

			ax[r, c].set_title(country_list[cols*r+c])

			ax[r, c].set_ylim(-100000000, 2500000000)

			ax[r, c].grid(c='Grey', alpha=1)

recs=[]

class_colour=['b','r','g']

for i in range(3):

	recs.append(patches.Rectangle((0,0),1,1,fc=class_colour[i]))

fig.legend(recs, ('Total Population', 'Pop under $1.9/d', 'Pop under $3.1/d'), loc=(0.815,0.92),

           fontsize=10)



country_list = ['China','India', 'Brazil', 'Argentina', 'Congo, Dem. Rep.', 'Zimbabwe']



pov_country_list = pov[pov['Country'].isin(country_list)]

for coun in country_list:

	pov_country = pov_country_list[pov_country_list['Country']==coun]

	print (coun, pov_country['ICode'].unique())



rows= 3

cols = 2



fig, ax = plt.subplots(rows, cols, figsize=(20,10))

plt.subplots_adjust(left=0.2, right=0.8, hspace=0.5)

for r in range(rows):

	for c in range(cols):

		pov_country = pov_country_list[pov_country_list['Country']==country_list[cols*r+c]]

		for year in range(1974, 2015):

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SP.POP.TOTL', str(year)],

                             c='b')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP1', str(year)]*1000000,

                             c='r')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP2', str(year)]*1000000,

                             c='g')

			ax[r, c].set_title(country_list[cols*r+c])

			ax[r, c].grid(c='Grey', alpha=1)

recs=[]

class_colour=['b','r','g']

for i in range(3):

	recs.append(patches.Rectangle((0,0),1,1,fc=class_colour[i]))

fig.legend(recs, ('Total Population', 'Pop under $1.9/d', 'Pop under $3.1/d'), loc=(0.815,0.92),

           fontsize=10)



import re



country_list = pov['Country'].unique()

country_list = [i for i in country_list if re.match('.*income*.', i, flags=re.IGNORECASE)]

country_list.remove('Low & middle income')



pov_country_list = pov[pov['Country'].isin(country_list)]

for coun in country_list:

	pov_country = pov_country_list[pov_country_list['Country']==coun]

	print (coun, pov_country['ICode'].unique())



rows= 2

cols = 2

fig, ax = plt.subplots(rows, cols, figsize=(18,10))

plt.subplots_adjust(left=0.2, right=0.8, hspace=0.5)

for r in range(rows):

	for c in range(cols):

		pov_country = pov_country_list[pov_country_list['Country']==country_list[cols*r+c]]

		for year in range(1974, 2015):

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SP.POP.TOTL', str(year)],

                             c='b')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP1', str(year)]*1000000,

                             c='r')

			ax[r, c].scatter(year, pov_country.loc[pov_country['ICode']=='SI.POV.NOP2', str(year)]*1000000,

                             c='g')

			ax[r, c].set_title(country_list[cols*r+c])

			ax[r, c].grid(c='Grey', alpha=1)

recs=[]

class_colour=['b','r','g']

for i in range(3):

	recs.append(patches.Rectangle((0,0),1,1,fc=class_colour[i]))

fig.legend(recs, ('Total Population', 'Pop under $1.9/d', 'Pop under $3.1/d'), loc=(0.85,0.85),

           fontsize=10)
