# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
from plotly.graph_objs import *

%matplotlib inline
sns.set()



df=pd.read_csv('../input/us-charities.csv') 
plotly.tools.set_credentials_file(username='vochiphat', api_key='aYSd242sROhvQX4Qbfgb')
df.info()
print(df.columns) 
df.head(5)
df.describe()
df.groupby('State')
df.groupby('State').groups
df.groupby(['State'])['Charity name'].count()
grouped = df.groupby('State')

for name, group in grouped:
    print(name)
    print(group)
for title, group in df.groupby('State'):
    group.hist(x='Charity name', y='Net assets', title=title)
texas=grouped.get_group('TX')
df['Net assets'].max()
data = [Bar(x=df["Charity name"],
            y=df["Net assets"])]

py.iplot(data,filename='bar_units sold')
df['Fundraising expenses'].max()
data = [Histogram(x=df['Fundraising expenses'])]


py.iplot(data,filename='histogram')
g= sns.FacetGrid(df, col='State')
g.map(plt.hist,'Administrative expenses');
g.add_legend
sns.set(rc={'figure.figsize':(110.7,80.27)})
sns.set_style("whitegrid")
sns.boxplot(x='Organization type', y="Net assets", data=df, palette="dark")
sns.despine(left=True)


