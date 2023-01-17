#Importing libraries
import pandas as pd 
import numpy as np
import xlrd
import inspect
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pandas import Timestamp
import seaborn as sns
#Importing data
Beja = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielbeja.xls')
Bizerte = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielbizerte.xls')
Gafsa = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielgafsa.xls')
Jendouba = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentieljendouba.xls')
Kairouan = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielkairouan.xls')
Kef = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielkef.xls')
Mahdia = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielmahdia.xls')
Mednine = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielmednine.xls')
Monastir = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielmonastir.xls')
Nabeul = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielnaeul.xls')
Sfax = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielsfax.xls')
SidiBouzid = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielsidibouzid.xls')
Siliana = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielsiliana.xls')
Sousse = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielsousse.xls')
Tataouine = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentieltataouine.xls')
Zaghouan = pd.read_excel('../input/prosl-elec-evolution-tatouine/evolutionprosolresidentielzaghouan.xls')
#Vector of 16 available gouvernorates
Tunisia = [Beja,Bizerte,Gafsa,Jendouba,Kairouan,Kef,Mahdia,Mednine,Monastir,Nabeul,Sfax,SidiBouzid,Siliana,Sousse,Tataouine,Zaghouan]

#Create a function that retrieve the name of any variable
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

#Create a function that a gouvernorate column 
def add_gouvernorate(Gov,name):
    Gov['Gouvernorate'] = np.repeat(name,len(Gov))
#Add 'Gouvernorate' column
Gov = Tunisia[0]
for Gov in Tunisia:
    name = retrieve_name(Gov)
    add_gouvernorate(Gov,name)
#Merge all data in one data frame 
df = pd.concat(Tunisia)
df.head(16)
#Line plot of the surface(m²) for each gouvernorate
    #NB : You can plot any other variable by switching y value
%matplotlib inline
matplotlib.rcParams['figure.figsize']=[12,10]
fig, ax = plt.subplots()
p = df.groupby('Gouvernorate').plot(x='Year', y='Surface m2', ax=ax)
plt.legend(np.unique(df['Gouvernorate']))
import plotly.plotly as py
import plotly.graph_objs as go

# Create traces
Gov = [Beja,Bizerte,Gafsa,Jendouba,Kairouan,Kef,Mahdia,Mednine,Monastir,Nabeul,Sfax,SidiBouzid,Siliana,Sousse,Tataouine,Zaghouan]
p = []
for i in range(1,len(Gov)):
    a = go.Scatter(
        x = Gov[i]['Year'].values,
        y = Gov[i]['Surface m2'].values,
        mode = 'lines+markers',
        name = retrieve_name(Gov[i]))
    p.append(a)
#py.iplot(p, filename='line-mode')
#Heatmap
corr = df[1:].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, 
            square=True,
            mask = mask ,
            vmax = .3,
            linewidths=.5,            
            cbar_kws={"shrink": .5},
            xticklabels=corr.columns,
            yticklabels=corr.columns)
