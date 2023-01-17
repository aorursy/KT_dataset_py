import pandas as pandasInstance
import numpy as numpyInstance
import seaborn as seabornInstance
import matplotlib.pyplot as matplotlibInstance
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
%matplotlib inline
init_notebook_mode(connected=True)
cf.go_offline()
pokemonDataFrame = pandasInstance.read_csv('../input/Pokemon.csv')
pokemonDataFrame.head()
pokemonDataFrame.info()
matplotlibInstance.figure(figsize=(30,25))
matplotlibInstance.tight_layout()
seabornInstance.set(font_scale=2)
seabornInstance.countplot(x='Type 1',data=pokemonDataFrame)
byLegendary = pokemonDataFrame.groupby(by='Legendary').sum()
byLegendaryTotal = byLegendary['Total']
matplotlibInstance.figure(figsize=(10,30))
matplotlibInstance.pie(byLegendaryTotal,labels=byLegendaryTotal.index,shadow=True)

matplotlibInstance.figure(figsize=(30,25))
matplotlibInstance.tight_layout()
pokemonDataFrame.iplot(kind='bar',x='Type 1',y='Total',color='purple')
matplotlibInstance.figure(figsize=(30,25))
matplotlibInstance.tight_layout()
pokemonDataFrame[['Name','HP']].iplot()
matplotlibInstance.figure(figsize=(30,25))
matplotlibInstance.tight_layout()
seabornInstance.countplot(x='Generation',data=pokemonDataFrame)
matplotlibInstance.figure(figsize=(30,25))
matplotlibInstance.tight_layout()
pokemonDataFrame.iplot(kind='bar',x='Type 1',y='Attack',color='red')
matplotlibInstance.figure(figsize=(30,25))
matplotlibInstance.tight_layout()
seabornInstance.boxplot(x='Type 1',y='Speed',data=pokemonDataFrame)

