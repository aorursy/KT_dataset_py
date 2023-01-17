import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as stats

import math

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline

import plotly.graph_objs as go
df = pd.read_csv('../input/2015.csv')
df.head(5)
df.info()
df.shape
df.columns
df.groupby('Region')['Happiness Rank', 'Happiness Score', 'Standard Error', 'Economy (GDP per Capita)'].mean().sort_values(by="Happiness Score", ascending=False)
plt.figure(figsize=(14,6))

topCountry=df.sort_values(by=['Happiness Rank'],ascending=True).head(15)

ax=sns.barplot(x='Country',y='Happiness Score', data=topCountry, palette = 'Set2')

ax.set(xlabel='Country', ylabel='Happiness Score')
plt.figure(figsize=(10,6))

list = df.sort_values(by=['Happiness Rank'],ascending=True)['Region'].head(30).value_counts()

list.plot(kind = 'bar', color = 'grey')
# Basic plotting on dataframe

df.plot(subplots=True, figsize=(12, 16))
sns.pairplot(df[['Happiness Score','Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom']])
df.hist(edgecolor = 'white', linewidth = 1, figsize = (20,16))

plt.show()
plt.figure(figsize=(10,8))

corr = df.drop(['Country','Region','Happiness Rank'],axis = 1).corr()

sns.heatmap(corr, cbar = True, square = True, annot=True, linewidths = .5, fmt='.2f',annot_kws={'size': 15}) 

sns.plt.title('Heatmap of Correlation Matrix')

plt.show()
sns.kdeplot(df['Happiness Score'], df['Economy (GDP per Capita)'], shade=True)

plt.scatter(df['Happiness Score'], df['Economy (GDP per Capita)'], alpha=0.2, color='green')

plt.xlabel('Happiness Score')

plt.ylabel('GDP per Capita')

plt.title('Happiness vs. GDP')

plt.show()





sns.kdeplot(df['Happiness Score'], df['Family'], shade=True)

plt.scatter(df['Happiness Score'], df['Family'], alpha=0.2, color='green')

plt.xlabel('Happiness Score')

plt.ylabel('Family')

plt.title('Family vs. GDP')

plt.show()





sns.kdeplot(df['Happiness Score'], df['Health (Life Expectancy)'], shade=True)

plt.scatter(df['Happiness Score'], df['Health (Life Expectancy)'], alpha=0.2, color='green')

plt.xlabel('Happiness Score')

plt.ylabel('Health (Life Expectancy)')

plt.title('Health (Life Expectancy) vs. GDP')

plt.show()





plt.figure(figsize=(12,12))

sns.jointplot(x = 'Economy (GDP per Capita)', y = 'Happiness Score', data = df, size=10, color='red')

plt.ylabel('Happiness Score', fontsize=12)

plt.xlabel('Economy (GDP per Capita)', fontsize=12)

plt.title('Economy (GDP per Capita) Vs Happiness', fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.regplot(x='Economy (GDP per Capita)',y='Happiness Score' ,data=df)

#so there is a linear relation between GDP & Happiness Score
plt.figure(figsize=(12,8))

sns.regplot(x='Generosity',y='Happiness Score' ,data=df)
cols = ['Standard Error', 'Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(cols):

    ax = plt.subplot(gs[i])

    #sns.distplot(df1[cn], bins=50)

    sns.regplot(x=df[cn],y='Happiness Score' ,data=df)

    ax.set_xlabel('')

    ax.set_title('Regrassion of feature: ' + str(cn))

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(df['Happiness Score'],kde=True, bins = 20)

plt.show()
#Percantage on country's regional listing 

df['Region'].value_counts().plot.pie(subplots=True, figsize=(8, 8), autopct='%.2f')
fig, axes = plt.subplots(figsize=(12, 6))

sns.boxplot(x='Region', y='Happiness Score', data = df)

plt.xticks(rotation=90)
order =['Sub-Saharan Africa', 'Southern Asia', 'Southeastern Asia', 'Eastern Asia', 'Australia and New Zealand', 'Central and Eastern Europe', 'Western Europe', 'Latin America and Caribbean', 'North America']



plt.figure(figsize=(10,6))

sns.barplot(x=df['Region'], y=df['Happiness Score'], order=order)

plt.xticks(rotation=75)

plt.xlabel('Regions')

plt.ylabel('Average Happiness Score 2015')

plt.title('Happiness by Region 2015')

plt.show()



plt.figure(figsize=(10,6))

sns.barplot(x=df['Region'], y=df['Economy (GDP per Capita)'], order=order)

plt.xticks(rotation=75)

plt.xlabel('Regions')

plt.ylabel('Average Economy (GDP per Capita) 2015')

plt.title('Economy (GDP per Capita) by Region 2015')

plt.show()



plt.figure(figsize=(10,6))

sns.barplot(x=df['Region'], y=df['Freedom'], order=order)

plt.xticks(rotation=75)

plt.xlabel('Regions')

plt.ylabel('Average Freedom 2015')

plt.title('Freedom by Region 2015')

plt.show()



plt.figure(figsize=(10,6))

sns.barplot(x=df['Region'], y=df['Family'], order=order)

plt.xticks(rotation=75)

plt.xlabel('Regions')

plt.ylabel('Average Family 2015')

plt.title('Family by Region 2015')

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="Region", y="Happiness Score", data=df, whis=np.inf)

sns.swarmplot(x="Region", y="Happiness Score",  data=df, split = True, palette='Set2', size = 6)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(12,6))

sns.stripplot(x="Region", y="Happiness Score", data=df, jitter=True)

plt.xticks(rotation=90)

plt.show()
w_europe = df[df.Region=='Western Europe']

ec_europe = df[df.Region=='Central and Eastern Europe']

europe = pd.concat([w_europe,ec_europe],axis=0)

europe.head()
plt.figure(figsize=(12,6))

sns.lmplot(data=europe, x='Economy (GDP per Capita)', y='Happiness Score', hue="Region")

plt.show()
selectCols=  ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Region']

sns.pairplot(europe[selectCols], hue='Region',size=2.5)
f, axes = plt.subplots(3, 2, figsize=(16, 16))

axes = axes.flatten()

compareCols = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']

for i in range(len(compareCols)):

    col = compareCols[i]

    axi = axes[i]

    sns.distplot(w_europe[col],color='blue' , label='West', ax=axi)

    sns.distplot(ec_europe[col],color='green', label='Central/East',ax=axi)

    axi.legend()
s_asia = df[df.Region=='Southern Asia']

e_asia = df[df.Region=='Eastern Asia']

se_asia = df[df.Region=='Southeastern Asia']



asia = pd.concat([s_asia, e_asia, se_asia],axis=0)

asia.head()
plt.figure(figsize=(12,8))

sns.lmplot(data=asia, x='Economy (GDP per Capita)', y='Happiness Score', hue="Region")

plt.show()
def plot_compare(dataset,regions,compareCols):

    n = len(compareCols)

    f, axes = plt.subplots(math.ceil(n/2), 2, figsize=(16, 6*math.ceil(n/2)))

    axes = axes.flatten()

    #compareCols = ['Happiness Score','Economy','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']

    for i in range(len(compareCols)):

        col = compareCols[i]

        axi = axes[i]

        for region in regions:

            this_region = dataset[dataset['Region']==region]

            sns.distplot(this_region[col], label=region, ax=axi)

        axi.legend()
regions = [

       'Middle East and Northern Africa', 'Latin America and Caribbean',

       'Southeastern Asia']

selectCol = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']

plot_compare(df,regions,selectCol)
regions = ['Western Europe', 'Middle East and Northern Africa',

       'Sub-Saharan Africa', 'Southern Asia']

selectCol = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']

plot_compare(df, regions, selectCol)
data = dict(type = 'choropleth', 

           locations = df['Country'],

           locationmode = 'country names',

           z = df['Happiness Rank'], 

           text = df['Country'],

           colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(120,120,120)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Happiness<br>Rank'),

      )



layout = dict(title = '2015 Global Happiness Index', 

             geo = dict(showframe = False, 

                        showcoastlines = False,

                       projection = {'type': 'Mercator'}))

choromap3 = dict(data = [data], layout=layout)

iplot(choromap3, validate = False, filename='Happiness-world-map-15')
X = df.drop(['Happiness Score', 'Happiness Rank', 'Country', 'Region'], axis=1)

y = df['Happiness Score']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print('Standardized features\n')

print(str(X_train[:4]))
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
result_lm = pd.DataFrame({

    'Actual':y_test,

    'Predict':y_pred

})

result_lm['Diff'] = y_test - y_pred

result_lm.head()
sns.regplot(x='Actual',y='Predict',data=result_lm)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
result_rf = pd.DataFrame({

    'Actual':y_test,

    'Predict':y_pred

})

result_rf['Diff'] = y_test - y_pred

result_rf.head()
plt.figure (figsize = (16, 8))

sns.pointplot(x='Actual',y='Predict',data=result_rf, dodge = True, color="#bb3f3f")

plt.xticks(rotation = 90)

plt.show()