import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly
import plotly.plotly as py
import plotly.offline as offline
from IPython.display import HTML
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
path1 = ("../input/world-happiness/2015.csv")
happiness2015 = pd.read_csv(path1)
happiness2015.head()
path2 = ('../input/world-happiness/2016.csv')
happiness2016 = pd.read_csv(path2)
happiness2016.head()
path3 = ('../input/world-happiness/2017.csv')
happiness2017 = pd.read_csv(path3)
happiness2017.head()
happinesseconomy  = happiness2017.groupby(['Country', 'Economy..GDP.per.Capita.']).size().to_frame('count').reset_index()
happinesseconomy = happinesseconomy.sort_values('Economy..GDP.per.Capita.', ascending=False)
happinesseconomy[['Country', 'Economy..GDP.per.Capita.']].head(10)
topHappiness = happiness2017[["Country","Happiness.Score"]]
topHappiness = topHappiness.sort_values(['Happiness.Score'], ascending=False).head(10)
topHappiness
leastHappiness = happiness2017[["Country","Happiness.Score"]]
leastHappiness = leastHappiness.sort_values(['Happiness.Score'], ascending=True).head(10)
leastHappiness
import seaborn as sns
import numpy as np
corr = happiness2017.corr()
sns.heatmap(corr, annot = True, cbar = True, fmt= '.2f', linewidths=.5)
happinesseconomy = happiness2017.groupby(['Country',"Happiness.Score",'Economy..GDP.per.Capita.',"Health..Life.Expectancy.","Family"]).size()
happinesseconomy = happinesseconomy.to_frame('count').reset_index()
happinesseconomy = happinesseconomy.sort_values('Happiness.Score', ascending=False)
happinesseconomy.head(5)

happinesseconomy.plot(kind='scatter', x='Economy..GDP.per.Capita.', y="Happiness.Score",alpha = 0.5,color = 'red')
plt.xlabel('Economy')              
plt.ylabel('Happiness')
plt.title('Economy with Happiness Score  Plot')
plt.show()
happinesseconomy.plot(kind='scatter', x= "Health..Life.Expectancy." , y="Happiness.Score",alpha = 0.5,color = 'blue')
plt.xlabel('Health')              
plt.ylabel('Happiness')
plt.title('Health with Happiness Score  Plot')
plt.show()
happinesseconomy.plot(kind='scatter', x= "Family" , y="Happiness.Score",alpha = 0.5,color = 'green')
plt.xlabel('Family')              
plt.ylabel('Happiness')
plt.title('Family with Happiness Score  Plot')
plt.show()
df2015 = happiness2015.rename(columns={'Happiness Score': 'Happiness Score15', 'Happiness Rank': 'Happiness Rank15'})
df2015 = df2015[["Country", 'Happiness Score15', 'Happiness Rank15']]
df2016 = happiness2016.rename(columns={'Happiness Score': 'Happiness Score16', 'Happiness Rank': 'Happiness Rank16'})
df2016 = df2016[["Country", 'Happiness Score16', 'Happiness Rank16']]
df2017 = happiness2017.rename(columns={'Happiness.Score': 'Happiness Score17', 'Happiness.Rank': 'Happiness Rank17'})
df2017 = df2017[["Country", 'Happiness Score17', 'Happiness Rank17']]
df3 = pd.merge(df2015, df2016)
dffinal = pd.merge(df3,df2017 )
dffinal.columns
dffinal["Diff"] = dffinal['Happiness Rank15']- dffinal['Happiness Rank17']
improving = dffinal.sort_values(['Diff'], ascending = False)
improving.head(5)
df2015 = happiness2015.rename(columns={'Happiness Score': 'Happiness Score15', 'Happiness Rank': 'Happiness Rank15'})
df2015 = df2015[["Country", 'Happiness Score15', 'Happiness Rank15']]
df2016 = happiness2016.rename(columns={'Happiness Score': 'Happiness Score16', 'Happiness Rank': 'Happiness Rank16'})
df2016 = df2016[["Country", 'Happiness Score16', 'Happiness Rank16']]
df2017 = happiness2017.rename(columns={'Happiness.Score': 'Happiness Score17', 'Happiness.Rank': 'Happiness Rank17'})
df2017 = df2017[["Country", 'Happiness Score17', 'Happiness Rank17']]
df3 = pd.merge(df2015, df2016)
dffinal = pd.merge(df3,df2017 )
dffinal.columns
dffinal["Diff"] = dffinal['Happiness Rank17']- dffinal['Happiness Rank15']
falling = dffinal.sort_values(['Diff'], ascending = False)
falling.head(5)
df15 = happiness2015.rename(columns={'Economy (GDP per Capita)': 'Economy (GDP per Capita)15', 'Health (Life Expectancy)': 'Health (Life Expectancy)15'})
df16 = happiness2016.rename(columns={'Economy (GDP per Capita)': 'Economy (GDP per Capita)16', 'Health (Life Expectancy)': 'Health (Life Expectancy)16'})
df17 = happiness2017.rename(columns={'Economy..GDP.per.Capita.': 'Economy..GDP.per.Capita.17', 'Health..Life.Expectancy.': 'Health..Life.Expectancy.17'})
df4 = pd.merge(df15,df16, on="Country")
dfcountry = pd.merge(df4,df17, on="Country")
dfnew = dfcountry.loc[dfcountry['Country'].isin(['Latvia','Egypt', 'Bulgaria','Hungary','Romania'])]
dfnew = dfnew[['Country','Economy (GDP per Capita)15','Economy (GDP per Capita)16','Economy..GDP.per.Capita.17','Health (Life Expectancy)15','Health (Life Expectancy)16','Health..Life.Expectancy.17']]
dfnew


df = happiness2015[['Region', 'Country']]
df1 = pd.merge(happiness2017,df, on="Country")
df1[df1.Region == 'Sub-Saharan Africa']
df015 = happiness2015.rename(columns={'Freedom': 'Freedom15', 'Trust (Government Corruption)': 'Government Corruption15'})
df016 = happiness2016.rename(columns={'Freedom': 'Freedom16', 'Trust (Government Corruption)': 'Government Corruption16'})
df017 = happiness2017.rename(columns={'Freedom': 'Freedom17', 'Trust..Government.Corruption.':'Government Corruption17'})
df04 = pd.merge(df015,df016, on="Country")
dfcountry = pd.merge(df04,df017, on="Country")
dfnew1 = dfcountry.loc[dfcountry['Country'].isin(['Venezuela','Liberia', 'Zambia','Haiti','Zimbabwe'])]
dfnew1 = dfnew1[['Country','Freedom15','Freedom16','Freedom17','Government Corruption15','Government Corruption16','Government Corruption17']]
dfnew1

path = ('../input/countries-iso-codes/wikipedia-iso-country-codes.csv')
df5 = pd.read_csv(path)
df6 = df5.rename(columns={'English short name lower case': 'Country'})
df7 = pd.merge(happiness2017,df6, on="Country")

data = [ dict(
        type = 'choropleth',
        locations = df7['Alpha-3 code'],
        z = df7['Happiness.Score'],
        text = df7['Country'],
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
            autotick = False,
            title = 'Happiness score'),
      ) ]

layout = dict(
    title = 'World Happiness',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data= data, layout=layout )
plotly.offline.iplot(fig, validate = False)

dystopia = happiness2017[['Dystopia.Residual', 'Happiness.Score']]
dystopia = dystopia.sort_values('Dystopia.Residual', ascending=False)
plt.plot(dystopia['Dystopia.Residual'],dystopia['Happiness.Score'])
plt.title("Dystopia Residual with Happiness Score")
plt.xlabel("Dystopia Residual")
plt.ylabel("Happiness Score")
plt.show

