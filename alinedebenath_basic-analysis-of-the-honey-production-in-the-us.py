import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly as py

import cufflinks as cf

cf.go_offline()

pd.set_option('display.max_columns', 30)
data = pd.read_csv("../input/honey-with-neonic-pesticide/vHoneyNeonic_v03.csv")
data.head()
data.shape
data.columns
data.insert(loc=3, column='yieldpercol_kg', value=data["yieldpercol"]*0.45359237)

data.insert(loc=5, column='totalprod_kg', value=data["totalprod"]*0.45359237)

data.insert(loc=6, column='totalprod_to', value=data["totalprod"]*0.00045359237)

data.insert(loc=8, column='stocks_to', value=data["stocks"]*0.00045359237)

data.insert(loc=10, column='priceperkg', value=data["priceperlb"]/0.45359237)

data.insert(loc=11, column='pricepertonne', value=data["priceperlb"]/0.00045359237)

data.head()
data = data.rename(columns={"nCLOTHIANIDIN": "CLOTHIANIDIN", "nIMIDACLOPRID": "IMIDACLOPRID",

                     "nTHIAMETHOXAM": "THIAMETHOXAM", "nACETAMIPRID": "ACETAMIPRID",

                    "nTHIACLOPRID": "THIACLOPRID","nAllNeonic":"AllNeonic"})

data.to_csv('vHoneyNeonic_v04.csv')
data.isnull().sum()
data = data.dropna()
data.shape
data.groupby("StateName")['totalprod_to'].sum().sort_values(ascending=False)[:10]
data.groupby("Region")['totalprod_kg'].sum().sort_values(ascending=False)
evo_price = data.groupby("year", as_index=False).agg({'priceperkg':'mean'})

evo_price.iplot(kind='line', x='year', xTitle='Year', color='orange',

           yTitle='Price of honey (dollars)', title='Evolution of the price of honey')
prod_by_year = data.groupby("year", as_index=False).agg({'totalprod_to':'mean'})

prod_by_year.iplot(kind='bar', x='year', xTitle='Year', color='red',

           yTitle='Production of honey (Tonne)', title='Evolution of the production of honey')
data['priceperkg'].corr(data['totalprod_kg'])
data.groupby("StateName")['AllNeonic'].sum().sort_values(ascending=False)
evo_neonic = data.groupby("year", as_index=False).agg({'AllNeonic':'mean'})

evo_neonic.iplot(kind='bar', x='year', xTitle='Year', color='green',

           yTitle='Use of Neonic pesticides (kg)', title='Evolution of the use of Neonic pesticides')
data['totalprod_kg'].corr(data['AllNeonic'])
evo_col = data.groupby("year", as_index=False).agg({'numcol':'count'})

evo_col.iplot(x='year', xTitle='Year', color='purple',

           yTitle='Number of colonies', title='Evolution of the number of colonies')
data['numcol'].corr(data['AllNeonic'])