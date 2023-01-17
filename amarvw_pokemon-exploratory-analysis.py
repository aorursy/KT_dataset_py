# This is my first attempt to create a kernal using kaggle. I have created some exploratory analysis based on the pokemon.csv datasource
# Most of the graphs are trivial, and my attempt to learn more about matplotlib, seaborn and plotly and cufflinks.
# import plotly.offline -> important since we are doing everything offline. For online, you would need a plotly account.
# TRY : remove the cf.go_offline() code and check the error, you'll get to know
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
# import the usual stack of libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style='darkgrid')
pokeman = pd.read_csv('../input/Pokemon.csv')
# check the pokemon data and try to explore more into it
pokeman.info()
pokeman.describe()
pokeman.dtypes
# Seaborn countplot to see the number of Legendary pokemons
sns.countplot(x='Legendary',data=pokeman)
sns.jointplot(x='HP',y='Attack',data=pokeman,kind='hex',color='g')
sns.pairplot(data=pokeman.drop(['Name','Type 2','Legendary','Generation','#'],axis=1),hue='Type 1')

pokeman[['HP','Attack']].iplot(kind='spread')
# Grouping the pokemon dataframe values for the mean of the given columns, I am using the new dataframe to create some charts
pokeman_gb = pokeman.groupby('Type 1')['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'].mean()
# reseting the index of the pokemon_gb dataframe, so that I can use the "Type 1" as a column name in my chart
pokeman_gb.reset_index(inplace=True)



pokeman_gb.iplot(x='Attack',y='Defense',mode='markers',xTitle='Attack',yTitle='Defense')
pokeman.plot.scatter(x='HP',y='Defense',c='Attack',s=pokeman['Total']/10.0,figsize=(12,12))
# interesting way to see the correlation between the columns in the dataframe, more the yellow, higher the correlation
sns.heatmap(data=pokeman_gb.corr(),cmap='plasma')
pokeman.plot.hexbin(x='HP',y='Attack',gridsize=25,cmap='coolwarm')
