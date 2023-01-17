!pip install chart_studio
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import os

import chart_studio.plotly as py  # plot package

import chart_studio.plotly

import numpy as np # linear algebra

import pandas as pd  # data-table package built upon numpy

import pycountry # lookup package for ISO-codes for countries of this blue, blue planet

from plotly.offline import iplot, init_notebook_mode  #iplot is for plotting into a jupyter

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
suspected_cases = pd.read_csv('/kaggle/input/meningococcal-meningitis-countrywise-data/Number of suspected meningitis cases reported.csv')

suspected_deaths = pd.read_csv('/kaggle/input/meningococcal-meningitis-countrywise-data/Number of suspected meningitis deaths reported.csv')

epidemic_districts = pd.read_csv('/kaggle/input/meningococcal-meningitis-countrywise-data/Number of meningitis epidemic districts.csv')
suspected_cases.head()
suspected_deaths.head()
epidemic_districts.head()
suspected_cases.shape
suspected_deaths.shape
epidemic_districts.shape
def clean_df(df):

    """

     - change here the year as the row index, and drop the "Country" row which has the year

     - change the index as int

     - reverse years ascending

     - replace "Not applicable" and "No data" values with NaN

     - change dtype to numeric

     - fill in NaN's with existing Data (interpolate "pad")  

    """

    df = df.T

    df.columns = df.iloc[0]

    df = df.iloc[1:]

    df.index = df["Country"]

    df = df.drop("Country", 1)

    df.index.names = ['Year']

    df.index = np.array(df.index).astype(int)

    df.columns.names = ["Country"]

    df = df.sort_index()

    

    

    df = df.replace("Not applicable", "NaN")

    df = df.replace("No data", "NaN")

    

    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.interpolate("pad")

    return df
suspected_cases = pd.read_csv('/kaggle/input/meningococcal-meningitis-countrywise-data/Number of suspected meningitis cases reported.csv')

suspected_deaths = pd.read_csv('/kaggle/input/meningococcal-meningitis-countrywise-data/Number of suspected meningitis deaths reported.csv')

epidemic_districts = pd.read_csv('/kaggle/input/meningococcal-meningitis-countrywise-data/Number of meningitis epidemic districts.csv')
suspected_cases = clean_df(suspected_cases)

suspected_deaths = clean_df(suspected_deaths)

epidemic_districts = clean_df(epidemic_districts)
suspected_cases
fig, axes = plt.subplots(1,3, figsize=(25, 10), dpi=200)

suspected_cases.plot.area(ax=axes[0], legend=True).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

suspected_deaths.plot.area(ax=axes[1], legend=True).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

epidemic_districts.plot.area(ax=axes[2], legend=True).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[0].set_title('suspected cases')

axes[1].set_title('suspected deaths')

axes[2].set_title('epidemic districs')

plt.tight_layout()
fig, axes = plt.subplots(1,3, figsize=(30,15), dpi=100)

suspected_cases.sum().plot.pie(y="Country", ax=axes[0])

suspected_deaths.sum().plot.pie(y="Country", ax=axes[1])

epidemic_districts.sum().plot.pie(y="Country", ax=axes[2])

axes[0].set_title('suspected cases')

axes[1].set_title('suspected deaths')

axes[2].set_title('epidemic districs')

axes[0].set_ylabel('') # remove None

axes[1].set_ylabel('') 

axes[2].set_ylabel('')

plt.tight_layout()
fig, axes = plt.subplots(3,1, figsize=(15,30), dpi=100)

suspected_deaths.plot.barh(ax=axes[0], rot=0).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[0].set_title('suspected_deaths')

suspected_cases.plot.barh(ax=axes[1], rot=0).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[1].set_title('suspected_cases')

epidemic_districts.plot.barh(ax=axes[2], rot=0).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[2].set_title('epidemic_districts')

plt.tight_layout()
fig, axes = plt.subplots(3,1, figsize=(12,15), dpi=100)

suspected_deaths.plot.box(ax=axes[0], rot=90).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[0].set_title('suspected_deaths')

suspected_cases.plot.box(ax=axes[1], rot=90).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[1].set_title('suspected_cases')

epidemic_districts.plot.box(ax=axes[2], rot=90).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))

axes[2].set_title('epidemic_districts')

plt.tight_layout()
def lookup(countries):

    result = []

    for i in range(len(countries)):

        try:

            result.append(pycountry.countries.get(name=countries[i]).alpha_3)

        except:

            try:

                result.append(pycountry.countries.get(official_name=countries[i]).alpha_3)

            except:

                result.append('undefined')

    return result
countries = suspected_deaths.T.index.values

codes=lookup(countries)

suspected_deaths = suspected_deaths.T

suspected_deaths["Codes"] = codes

suspected_deaths=suspected_deaths[~suspected_deaths.Codes.isin(['undefined'])]
countries = suspected_cases.T.index.values

codes=lookup(countries)

suspected_cases = suspected_cases.T

suspected_cases["Codes"] = codes

suspected_cases=suspected_cases[~suspected_cases.Codes.isin(['undefined'])]
countries = epidemic_districts.T.index.values

codes=lookup(countries)

epidemic_districts = epidemic_districts.T

epidemic_districts["Codes"] = codes

epidemic_districts=epidemic_districts[~epidemic_districts.Codes.isin(['undefined'])]
def get_data_layout_map(df_, cbar_label='sum over previous years', title_plot='epidemic_district'):

    df = pd.DataFrame(df_.iloc[:,:-1].sum(axis=1))

    df.columns = ['sum']

    df["Codes"] = df_.iloc[:,-1]



    data = [ dict(

        type = 'choropleth',

        locations = df['Codes'],

        z = df['sum'],

        text = df.index,

       colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"], \

                     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

       autocolorscale = False,

       reversescale = True,

        marker = dict(

           line = dict (

               color = 'rgb(180,180,180)',

               width = 0.5

           ) ),

        colorbar = dict(

           autotick = False,

           title = cbar_label),

    ) ]





    layout = dict(

        title = title_plot,

        geo = dict(

            showframe = False,

            showcoastlines = False,

            projection = dict(

                type = 'Mercator'

            )

        )

    )

    return data, layout
data, layout = get_data_layout_map(epidemic_districts, cbar_label='sum over previous years', title_plot='epidemic_districts 2003-2014')

fig = dict( data=data, layout=layout )

iplot(fig, validate=False)
data, layout = get_data_layout_map(suspected_cases, cbar_label='sum over previous years', title_plot='suspected_cases 2009-2014')

fig = dict( data=data, layout=layout )

iplot(fig, validate=False)
data, layout = get_data_layout_map(suspected_deaths, cbar_label='sum over previous years', title_plot='suspected_deaths 1965-2014')

fig = dict( data=data, layout=layout )

iplot(fig, validate=False)