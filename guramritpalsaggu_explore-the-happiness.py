import pandas as pd # data processing

import chart_studio.plotly as py #for data visualization

import plotly.graph_objs as go #for data visualization

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import seaborn as sns #for data visualization

import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = (20, 10)
df_2015 = pd.read_csv('../input/world-happiness/2015.csv')

df_2016 = pd.read_csv('../input/world-happiness/2016.csv')

df_2017 = pd.read_csv('../input/world-happiness/2017.csv')
df_2015.describe()
df_2015.columns
df_2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',

       'Standard Error', 'Economy', 'Family',

       'Health', 'Freedom', 'Trust',

       'Generosity', 'Dystopia_Residual']

new_df_2015 = df_2015.drop(['Standard Error'], axis=1)
new_df_2015.head()
drop_2016 = ['Lower Confidence Interval','Upper Confidence Interval' ]

new_df_2016 = df_2016.drop(drop_2016, axis=1)

new_df_2016.columns = ['Country', 'Region','Happiness_Rank', 'Happiness_Score','Economy', 'Family',

       'Health', 'Freedom', 'Trust',

       'Generosity', 'Dystopia_Residual']
new_df_2016.head()
columns_2017 = ['Whisker.high','Whisker.low' ]

new_df_2017 = df_2017.drop(columns_2017, axis=1)

new_df_2017.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',

       'Health', 'Freedom', 'Trust',

       'Generosity', 'Dystopia_Residual']
new_df_2017.head()
new_df_2015['Year']=2015

new_df_2016['Year']=2016

new_df_2017['Year']=2017

frames = [new_df_2015, new_df_2016, new_df_2017]

happiness = pd.concat(frames,sort=True)
happiness.head()

new_df_2016.head()
data1 = dict(type = 'choropleth', 

           locations = happiness['Country'],

           locationmode = 'country names',

           z = happiness['Happiness_Rank'], 

           text = happiness['Country'],

          colorscale = 'Viridis', reversescale = False)

layout = dict(title = 'Happiness Rank Across the World', 

             geo = dict(showframe = False, 

                       projection = {'type': 'mercator'}))

choromap6 = go.Figure(data = [data1], layout=layout)

iplot(choromap6)
data2 = dict(type = 'choropleth', 

           locations = happiness['Country'],

           locationmode = 'country names',

           z = happiness['Happiness_Score'], 

           text = happiness['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Happiness Score Across the World', 

             geo = dict(showframe = False, 

                       projection = {'type': 'mercator'}))

choromap3 = go.Figure(data = [data2], layout=layout)

iplot(choromap3)
f,ax = plt.subplots(figsize =(20,10))

sns.boxplot(x="Year" , y="Happiness_Score", hue="Region",data=happiness,palette="PRGn",ax=ax)

plt.show()
sns.heatmap(new_df_2015.corr(), cmap='Blues',annot = True)

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
sns.heatmap(new_df_2016.corr(), cmap='Blues',annot = True)

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
data4 = new_df_2015.groupby('Region')['Happiness_Score','Economy','Family','Health'].median()

data4 = pd.DataFrame(data4)

data4
sns.scatterplot(data4['Happiness_Score'], data4['Economy'],hue = data4.index, legend='brief',s=200)

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
sns.scatterplot(data4['Happiness_Score'], data4['Family'],hue = data4.index, legend='brief',s=200)

plt.show()
sns.scatterplot(data4['Happiness_Score'], data4['Health'],hue = data4.index, legend='brief',s=200)

plt.show()
df_1 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Western Europe']

df_2 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'North America']

df = pd.concat([df_1, df_2], axis = 0)

sns.heatmap(df.corr(), cmap = 'Blues', annot = True)

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
plt.subplot(1,2,1)

sns.scatterplot(new_df_2016['Happiness_Score'], new_df_2016['Economy'],)

plt.subplot(1,2,2)

sns.scatterplot(new_df_2016['Happiness_Score'], new_df_2016['Health'])

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
df_1 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Eastern Asia']

df_2 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Sub Saharan Africa']

df_3 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Southern Asia']

df = pd.concat([df_1, df_2,df_3], axis = 0)

sns.heatmap(df.corr(), cmap = 'Blues', annot = True)

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
plt.subplot(1,3,1)

sns.scatterplot(df['Economy'],df['Happiness_Score'])

plt.subplot(1,3,2)

sns.scatterplot(df['Family'],df['Happiness_Score'])

plt.subplot(1,3,3)

sns.scatterplot(df['Health'],df['Happiness_Score'])

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()
df_1 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Australia and New Zealand']

df_2 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Middle East and Northern Africa']

df_3 = new_df_2016.loc[lambda new_df_2016: new_df_2016['Region'] == 'Latin America and Caribean']

df = pd.concat([df_1, df_2,df_3], axis = 0)

sns.heatmap(df.corr(), cmap = 'Blues', annot = True)

plt.rcParams['figure.figsize'] = (20, 10)

plt.show()