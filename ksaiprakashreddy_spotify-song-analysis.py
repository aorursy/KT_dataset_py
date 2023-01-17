import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
df=pd.read_csv("../input/top50spotify2019/top50.csv",encoding="ISO-8859-1")
df.head()
fig = px.treemap(df,path=['Genre', 'Artist.Name'],values=df['Popularity'])
fig.show()
df.drop(['Unnamed: 0'],axis = 1, inplace = True)
relation = df.groupby(['Genre', 'Artist.Name']).sum()
relation
plt.figure(figsize = (10,7))
sns.set_style("whitegrid")
sns.distplot(df['Popularity'])
plt.show()
plt.figure(figsize=(10,7))
sns.distplot(df['Danceability'], kde=False, bins=15,color='red')
plt.show()
# heatmap of the correlation 

correlation=df.corr(method='pearson')
plt.figure(figsize=(10,10))
plt.title('Correlation heatmap',fontsize=20)
sns.heatmap(correlation,annot=True, center = 1)
plt.show()
