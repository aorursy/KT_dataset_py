import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



df = pd.read_csv('../input/2015.csv')



df.head()
df.tail()
df.info()
df['Country'].nunique()
df['Region'].nunique()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline



dfMap = dict(type = 'choropleth', locations = df['Country'],locationmode = 'country names',z = df['Happiness Score'], text = df['Country'],colorbar = {'title':'World Happiness Score'})

layout = dict(title = 'World Happiness Score', geo = dict(showframe = False))

WorldMap = go.Figure(data = [dfMap], layout=layout)

iplot(WorldMap)
#Its very curious the STD

dfSTD = df.sort_values(by='Standard Error', axis=0, ascending=False)

dfSTD.head(10)
dfCorr = df.drop(['Happiness Rank'],axis=1)
plt.subplots(figsize=(15,10))

sns.heatmap(dfCorr.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)

plt.title('Correlation of the dataset', size=25)

plt.show()
sns.jointplot(x='Happiness Score',y='Economy (GDP per Capita)',data=df, kind="reg",height=10, color="b")

sns.jointplot(x='Happiness Score',y='Family',data=df, kind="reg",height=10, color="b")

sns.jointplot(x='Happiness Score',y='Health (Life Expectancy)',data=df, kind="reg",height=10, color="b")
df['Continent'] = df['Region']
df['Continent']=df['Continent'].apply(lambda x: x.replace('Western Europe', 'Europe'))

df['Continent']=df['Continent'].apply(lambda x: x.replace('Central and Eastern Europe', 'Europe'))

df['Continent']=df['Continent'].apply(lambda x: x.replace('Middle East and Northern Africa', 'Africa'))

df['Continent']=df['Continent'].apply(lambda x: x.replace('Sub-Saharan Africa', 'Africa'))

df['Continent']=df['Continent'].apply(lambda x: x.replace('Southeastern Asia', 'Asia'))

df['Continent']=df['Continent'].apply(lambda x: x.replace('Eastern Asia', 'Asia'))

df['Continent']=df['Continent'].apply(lambda x: x.replace('Southern Asia', 'Asia'))
df['Continent'].unique()
df.head()
dfCorrEurope = df.loc[df['Continent']=='Europe']

dfCorrEurope = dfCorrEurope.drop(['Happiness Rank'],axis=1)

plt.subplots(figsize=(15,10))

sns.heatmap(dfCorrEurope.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)

plt.title('Correlation of the dataset Europe', size=25)

plt.show()

dfCorrAfrica = df.loc[df['Continent']=='Africa']

dfCorrAfrica = dfCorrAfrica.drop(['Happiness Rank'],axis=1)

plt.subplots(figsize=(15,10))

sns.heatmap(dfCorrAfrica.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)

plt.title('Correlation of the dataset Africa', size=25)

plt.show()

dfCorrSA = df.loc[df['Continent']=='Latin America and Caribbean']

dfCorrSA = dfCorrSA.drop(['Happiness Rank'],axis=1)

plt.subplots(figsize=(15,10))

sns.heatmap(dfCorrSA.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)

plt.title('Correlation of the dataset South America', size=25)

plt.show()
dfCorrAsia = df.loc[df['Continent']=='Asia']

dfCorrAsia = dfCorrAsia.drop(['Happiness Rank'],axis=1)

plt.subplots(figsize=(15,10))

sns.heatmap(dfCorrAsia.corr(), annot=True, linewidths = 0.2, linecolor='black', center=1)

plt.title('Correlation of the dataset Asia', size=25)

plt.show()
sns.set(rc={'figure.figsize':(18,9)})



sns.boxplot(x="Continent", y="Happiness Score", data=df)
sns.swarmplot(x="Continent", y="Happiness Score", data=df,palette="Set2", dodge=True)


dfTop = df.head(12)

sns.barplot(x="Country", y="Happiness Score", data=dfTop, ci=68,  palette="Blues_d")
dfPerContinent = df.groupby(by='Continent')['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)'

                           ,'Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual'].mean()

dfPerContinent=dfPerContinent.reset_index()



dfPerContinentO = dfPerContinent.sort_values(by='Happiness Score',ascending=False)

sns.barplot(x="Continent", y="Happiness Score", data=dfPerContinentO, ci=68,  palette="Blues_d")
dfPerRegion = df.groupby(by='Region')['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)'

                           ,'Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual'].mean()



dfPerRegionO = dfPerRegion.sort_values(by='Happiness Score',ascending=False)

dfPerRegionO=dfPerRegionO.reset_index()

g = sns.barplot(x="Region", y="Happiness Score", data=dfPerRegionO, ci=68,  palette="Blues_d")

plt.xticks(rotation=90)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



df1 = df.drop(columns=['Country','Happiness Rank','Region'])



X = df1.drop(columns=['Happiness Score'])

y = df1['Happiness Score']



X1 = np.array(X)

y1 = np.array(y)



X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state=42)



reg_all = LinearRegression()



reg_all.fit(X_train, y_train)



y_pred = reg_all.predict(X_test)



print("R^2: {}".format(reg_all.score(X_test, y_test)))


