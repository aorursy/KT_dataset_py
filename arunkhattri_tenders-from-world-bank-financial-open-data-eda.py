# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/procurement-notices/procurement-notices.csv", parse_dates=['Publication Date', 'Deadline Date'])
df.columns
df.info()
# Count of Major Sector's in Data
df['Major Sector'].nunique()
# How many notice type data has ?
df['Notice Type'].nunique()
# Notice Types
df['Notice Type'].unique()
start = df['Publication Date'].min()
end = df['Publication Date'].max()
print(f"latest date in data: {end}")
print(f"oldest date in data: {start}")
# since start till end each day how many contracts were awarded ?
date_of_interest = pd.date_range(start=start, end=end)

def count_contract_awarded(dates):
    count = len(df.loc[(df['Notice Type'] == 'Contract Award') & (df['Publication Date'] == dates),:].index)
    return count

contract_awarded = []
for date in date_of_interest:
    contract_awarded.append(count_contract_awarded(date))


# plt.bar(date_of_interest, contract_awarded)
# plt.xlabel('Dates')
# plt.ylabel("Counts of Contract Awarded")
# plt.xticks(rotation=90)
# plt.show()

data = [go.Bar(
            x=date_of_interest,
            y=contract_awarded
    )]

# py.iplot(data, filename='basic-bar')

# specify the layout of our figure
layout = dict(title = "Contracts Awarded by World Bank",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
# Count of Tenders currently out
today = datetime.today().strftime('%Y, %m, %d')
open_tenders = len(df.loc[df['Deadline Date'] > today,:].index)
print(f"Open Tenders: {open_tenders}")
# countries associated with award of contract
df_contract = df.loc[(df['Notice Type'] == 'Contract Award'),:]
df_contract.head()
# df_contract.groupby(['Publication Date', 'Country Name']).count()
countries_awards = df_contract.groupby('Country Name').count()
countries_awards
country_count_of_contracts_award = countries_awards.iloc[:,0]
country_count_of_contracts_award.head()
country_count_of_award = pd.DataFrame({'COUNTRY':country_count_of_contracts_award.index,
                                       'count':country_count_of_contracts_award.values})

type(country_count_of_award)
country_count_of_award.head()
# country codes for plotly
# as in the data some country codes are missing
df_codes = pd.read_csv("../input/country-codes-for-plotly/plotlyCountriesCodes.csv")
df_codes.head()
# inner join data sets
frames = [country_count_of_contracts_award, df_codes]
df_country_awards = pd.merge(left=country_count_of_award,
                             right = df_codes,
                             on='COUNTRY')
df_country_awards.head()
data = [dict(
type = 'choropleth',
locations = df_country_awards['CODE'],
z = df_country_awards['count'],
text = df_country_awards['COUNTRY'],
colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
autocolorscale = False,
reversescale = True,
marker = dict(
line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '#',
            title = 'Contracts Awarded #'),
)]

layout = dict(
    title = 'Contracts Awarded by World Bank',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig, filename='countries-contract-awarded-map')

