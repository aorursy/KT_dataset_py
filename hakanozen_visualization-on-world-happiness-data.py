import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')
df2015 = pd.read_csv("../input/2015.csv")
df2016 = pd.read_csv("../input/2016.csv")
df2017 = pd.read_csv("../input/2017.csv")
df2015.head()
# Drop columns that will not be used in further analysis
df2015.drop(["Standard Error"], axis = 1, inplace = True)
df2016.head()
# Drop columns that will not be used in further analysis
df2016.drop(["Lower Confidence Interval"], axis = 1, inplace = True)
df2016.drop(["Upper Confidence Interval"], axis = 1, inplace = True)
df2017.head()
# Drop columns that will not be used in further analysis
df2017.drop(["Whisker.high"], axis = 1, inplace = True)
df2017.drop(["Whisker.low"], axis = 1, inplace = True)
print(df2015.shape)
print(df2016.shape)
print(df2017.shape)
country2015 = df2015.Country.tolist() # series to list conversion
country2016 = df2016.Country.tolist() # series to list conversion
country2017 = df2017.Country.tolist() # series to list conversion
countries = country2015 + country2016 + country2017 # union of the countries

countries = list(set(countries)) # remove duplicates, end up with unique elements

print("countries[] contains " + str(len(countries)) + " unique elements before removing the common elements")
uncommon_countries = countries.copy() # copy operation is important. Otherwise, 
                                      # modifications on uncommon_countries[] change countries[] too.
for country in uncommon_countries:
    if country in country2015 and country in country2016 and country in country2017:
        countries.remove(country)
        
uncommon_countries = countries.copy()        
print("uncommon_countries[] contains " + str(len(uncommon_countries)) + " unique elements after removing the common elements")
for i in range(0,len(df2015.Country)):
    if df2015.Country[i] in uncommon_countries:
        df2015.drop(i, axis = 0, inplace = True)
for i in range(0,len(df2016.Country)):
    if df2016.Country[i] in uncommon_countries:
        df2016.drop(i, axis = 0, inplace = True)
for i in range(0,len(df2017.Country)):
    if df2017.Country[i] in uncommon_countries:
        df2017.drop(i, axis = 0, inplace = True)
    
print(df2015.shape)
print(df2016.shape)
print(df2017.shape)
df2015.reset_index(inplace = True)
df2016.reset_index(inplace = True)
df2017.reset_index(inplace = True)
# order countries alphabetically so that countries in dataframes have the same order
df2015.sort_values("Country", inplace = True)
df2016.sort_values("Country", inplace = True)
df2017.sort_values("Country", inplace = True)
df2017.insert(1, 'Region', df2017['Country']) # create "region" column as the 2nd column of df2017

# fill the "Region" column created in df2017
for i in range(0,df2017.shape[0]):
    df2017.Region[i] = df2015.Region[i] # this works since countries are in the same order
    df2016.Region[i] = df2015.Region[i] # this works since countries are in the same order
# re-order dataframes according to the index
df2015.sort_index(inplace=True)
df2016.sort_index(inplace=True)
df2017.sort_index(inplace=True)

# drop index column that is additionally created for sorting (ordering)
df2015.drop(['index'], axis = 1, inplace = True)
df2016.drop(['index'], axis = 1, inplace = True)
df2017.drop(['index'], axis = 1, inplace = True)

# re-order column names in alphabetical order
df2015 = df2015.reindex_axis(sorted(df2015.columns), axis=1)
df2016 = df2016.reindex_axis(sorted(df2016.columns), axis=1)
df2017 = df2017.reindex_axis(sorted(df2017.columns), axis=1)
df2015.rename(index=str, columns={"Country": "country", 
                                  "Dystopia Residual": "dystopia_residual",
                                  "Economy (GDP per Capita)": "GDP_per_Capita",
                                  "Family": "family",
                                  "Freedom": "freedom",
                                  "Generosity": "generosity",
                                  "Happiness Rank": "happiness_rank",
                                  "Happiness Score": "happiness_score",
                                  "Health (Life Expectancy)": "health",
                                  "Region": "region",
                                  "Trust (Government Corruption)": "trust_to_gov"}, inplace = True)

df2016.rename(index=str, columns={"Country": "country", 
                                  "Dystopia Residual": "dystopia_residual",
                                  "Economy (GDP per Capita)": "GDP_per_Capita",
                                  "Family": "family",
                                  "Freedom": "freedom",
                                  "Generosity": "generosity",
                                  "Happiness Rank": "happiness_rank",
                                  "Happiness Score": "happiness_score",
                                  "Health (Life Expectancy)": "health",
                                  "Region": "region",
                                  "Trust (Government Corruption)": "trust_to_gov"}, inplace = True)

df2017.rename(index=str, columns={"Country": "country", 
                                  "Dystopia.Residual": "dystopia_residual",
                                  "Economy..GDP.per.Capita.": "GDP_per_Capita",
                                  "Family": "family",
                                  "Freedom": "freedom",
                                  "Generosity": "generosity",
                                  "Happiness.Rank": "happiness_rank",
                                  "Happiness.Score": "happiness_score",
                                  "Health..Life.Expectancy.": "health",
                                  "Region": "region",
                                  "Trust..Government.Corruption.": "trust_to_gov"}, inplace = True)
df2015.head()
df2016.head()
df2017.head()
## Top 10 Happiest Countries

happiness_score_2015 = []
countries_2015 = []
happiness_score_2016 = []
countries_2016 = []
happiness_score_2017 = []
countries_2017 = []

for i in range(0,10):
    happiness_score_2015.append(df2015.happiness_score.values[i])
    countries_2015.append(df2015.country[i])
    happiness_score_2016.append(df2016.happiness_score.values[i])
    countries_2016.append(df2016.country[i])
    happiness_score_2017.append(df2017.happiness_score.values[i])
    countries_2017.append(df2017.country[i])

# Normalize happiness score by the maximum happiness score at the associated year
happiness_score_2015 = happiness_score_2015/happiness_score_2015[0]*100
happiness_score_2016 = happiness_score_2016/happiness_score_2016[0]*100
happiness_score_2017 = happiness_score_2017/happiness_score_2017[0]*100

# create trace1   
trace1 = go.Bar(
    x = countries_2015,
    y = happiness_score_2015,
    name = "2015")
trace2 = go.Bar(
    x = countries_2016,
    y = happiness_score_2016,
    name = "2016")
trace3 = go.Bar(
    x = countries_2017,
    y = happiness_score_2016,
    name = "2017")

data = [trace1, trace2, trace3]
layout = go.Layout(
    margin=dict(b=100),
    title='Top 10 Happiest Countries',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd', range=[95, 100]),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
## Top 10 Saddest Countries

happiness_score_2015 = []
countries_2015 = []
happiness_score_2016 = []
countries_2016 = []
happiness_score_2017 = []
countries_2017 = []

for i in range(1,9):
    happiness_score_2015.append(df2015.happiness_score.values[-i])
    countries_2015.append(df2015.country[-i])
    happiness_score_2016.append(df2016.happiness_score.values[-i])
    countries_2016.append(df2016.country[-i])
    happiness_score_2017.append(df2017.happiness_score.values[-i])
    countries_2017.append(df2017.country[-i])

# Normalize happiness score by the maximum happiness score at the associated year
happiness_score_2015 = happiness_score_2015/df2015.happiness_score.values[0]*100
happiness_score_2016 = happiness_score_2016/df2016.happiness_score.values[0]*100
happiness_score_2017 = happiness_score_2017/df2017.happiness_score.values[0]*100

# create trace1   
trace1 = go.Bar(
    x = countries_2015,
    y = happiness_score_2015,
    name = "2015")
trace2 = go.Bar(
    x = countries_2016,
    y = happiness_score_2016,
    name = "2016")
trace3 = go.Bar(
    x = countries_2017,
    y = happiness_score_2016,
    name = "2017")

data1 = [trace1]
data2 = [trace2]
data3 = [trace3]
layout1 = go.Layout(
    margin=dict(b=100),
    title='Top 10 Saddest Countries in 2015',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

layout2 = go.Layout(
    margin=dict(b=100),
    title='Top 10 Saddest Countries in 2016',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

layout3 = go.Layout(
    margin=dict(b=100),
    title='Top 10 Saddest Countries in 2017',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

fig1 = go.Figure(data=data1, layout=layout1)
fig2 = go.Figure(data=data2, layout=layout2)
fig3 = go.Figure(data=data3, layout=layout3)
py.iplot(fig1, filename='grouped-bar')
py.iplot(fig2, filename='grouped-bar')
py.iplot(fig3, filename='grouped-bar')
## Average Happiness Scores of the Regions

regions = df2015.region.tolist()
regions = list(set(regions)) # returns the list of unique regions

region2015 = {}
region2016 = {}
region2017 = {}

for i in range(0, len(df2015.region)):
    if df2015.region[i] not in region2015:
        region2015[df2015.region[i]] = [df2015.happiness_score[i]]
    else:
        region2015[df2015.region[i]].append(df2015.happiness_score[i])
        
    if df2016.region[i] not in region2016:
        region2016[df2016.region[i]] = [df2016.happiness_score[i]]
    else:
        region2016[df2016.region[i]].append(df2016.happiness_score[i])
        
    if df2017.region[i] not in region2017:
        region2017[df2017.region[i]] = [df2017.happiness_score[i]]
    else:
        region2017[df2017.region[i]].append(df2017.happiness_score[i])

region_2015 = []
region_2016 = []
region_2017 = []

for region in regions:
    avg2015 = sum(region2015[region])/len(region2015[region])
    region_2015.append(avg2015)
    
    avg2016 = sum(region2016[region])/len(region2016[region])
    region_2016.append(avg2016)
    
    avg2017 = sum(region2017[region])/len(region2017[region])
    region_2017.append(avg2017)

# Normalize happiness scores of regions by the greatest score at that year
region_2015 = region_2015/df2015.happiness_score.values[0]*100
region_2016 = region_2016/df2016.happiness_score.values[0]*100
region_2017 = region_2017/df2017.happiness_score.values[0]*100

# create trace1   
trace1 = go.Bar(
    x = regions,
    y = region_2015,
    name = "2015")
trace2 = go.Bar(
    x = regions,
    y = region_2016,
    name = "2016")
trace3 = go.Bar(
    x = regions,
    y = region_2017,
    name = "2017")

data = [trace1, trace2, trace3]
layout = go.Layout(
    margin=dict(b=250),
    title='Average Happiness Scores of the Regions',
    xaxis=dict(titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='Happiness Scores (%)',gridwidth=2, gridcolor='#bdbdbd', range = [50, 100]),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')

## Contribution Levels of the Factors to the Happiness

factors_2015 = []
contribution_of_factors_2015 = []
factors_2016 = []
contribution_of_factors_2016 = []
factors_2017 = []
contribution_of_factors_2017 = []

for col_name in list(df2015.columns.values):
    if not (col_name == 'country' or col_name == 'happiness_rank' or col_name == 'region' or col_name == 'happiness_score'):
        factor_2015 = df2015[col_name].values
        #print(factor_2015)
        avg_contribution_2015 = sum(factor_2015)/len(factor_2015)
        factors_2015.append(col_name)
        contribution_of_factors_2015.append(avg_contribution_2015)
    
        factor_2016 = df2016[col_name].values
        avg_contribution_2016 = sum(factor_2016)/len(factor_2016)
        factors_2016.append(col_name)
        contribution_of_factors_2016.append(avg_contribution_2016)
    
        factor_2017 = df2017[col_name].values
        avg_contribution_2017 = sum(factor_2017)/len(factor_2017)
        factors_2017.append(col_name)
        contribution_of_factors_2017.append(avg_contribution_2017)
        
contribution_of_factors_2015 = contribution_of_factors_2015/sum(contribution_of_factors_2015) * 100
contribution_of_factors_2016 = contribution_of_factors_2016/sum(contribution_of_factors_2016) * 100
contribution_of_factors_2017 = contribution_of_factors_2017/sum(contribution_of_factors_2017) * 100

# create trace1   
trace1 = go.Bar(
    x = factors_2015,
    y = contribution_of_factors_2015,
    name = "2015")
trace2 = go.Bar(
    x = factors_2016,
    y = contribution_of_factors_2016,
    name = "2016")
trace3 = go.Bar(
    x = factors_2017,
    y = contribution_of_factors_2017,
    name = "2017")

data = [trace1, trace2, trace3]
layout = go.Layout(
    margin=dict(b=150),
    title='Contribution Levels of the Factors to the Happiness',
    xaxis=dict(title='Factors',titlefont=dict(size=16), tickangle=-60),
    yaxis=dict(title='Contribution Levels (%)',gridwidth=2, gridcolor='#bdbdbd'),
    font=dict(size=16),
    bargap = 0.6,
    barmode='group')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
## Contribution Weight of Dystopia Residual to Happiness

dystopia_residual2015 = df2015.dystopia_residual.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
dystopia_residual2016 = df2016.dystopia_residual.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
dystopia_residual2017 = df2017.dystopia_residual.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 30

for i in range(0,len(dystopia_residual2015)):
    dystopia_residual2015[i] = sum(dystopia_residual2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    dystopia_residual2016[i] = sum(dystopia_residual2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    dystopia_residual2017[i] = sum(dystopia_residual2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100

trace1 = go.Scatter(
                    y = dystopia_residual2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = dystopia_residual2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = dystopia_residual2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of Dystopia Residual to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of Dystopia Residual (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
## Contribution Weight of GDP per Capita

GDP2015 = df2015.GDP_per_Capita.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
GDP2016 = df2016.GDP_per_Capita.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
GDP2017 = df2017.GDP_per_Capita.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 30

for i in range(0,len(GDP2015)):
    GDP2015[i] = sum(GDP2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    GDP2016[i] = sum(GDP2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    GDP2017[i] = sum(GDP2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100

trace1 = go.Scatter(
                    y = GDP2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = GDP2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = GDP2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of GDP per Capita to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of GDP per Capita (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
## Contribution Weight of Family

family2015 = df2015.family.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
family2016 = df2016.family.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
family2017 = df2017.family.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 20

for i in range(0,len(family2015)):
    family2015[i] = sum(family2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    family2016[i] = sum(family2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    family2017[i] = sum(family2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100    

trace1 = go.Scatter(
                    y = family2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = family2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = family2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of Family to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of Family (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
## Contribution Weight of Freedom

freedom2015 = df2015.freedom.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
freedom2016 = df2016.freedom.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
freedom2017 = df2017.freedom.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 40

for i in range(0,len(dystopia_residual2015)):
    freedom2015[i] = sum(freedom2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    freedom2016[i] = sum(freedom2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    freedom2017[i] = sum(freedom2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100
    
trace1 = go.Scatter(
                    y = freedom2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = freedom2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = freedom2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of Freedom to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of Freedom (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
## Contribution Weight of Generosity

generosity2015 = df2015.generosity.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
generosity2016 = df2016.generosity.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
generosity2017 = df2017.generosity.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 30

for i in range(0,len(dystopia_residual2015)):
    generosity2015[i] = sum(generosity2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    generosity2016[i] = sum(generosity2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    generosity2017[i] = sum(generosity2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100    
    
trace1 = go.Scatter(
                    y = generosity2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = generosity2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = generosity2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of Generosity to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of Generosity (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
## Contribution Weight of Health

health2015 = df2015.health.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
health2016 = df2016.health.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
health2017 = df2017.health.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 30

for i in range(0,len(dystopia_residual2015)):
    health2015[i] = sum(health2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    health2016[i] = sum(health2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    health2017[i] = sum(health2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100    
    
trace1 = go.Scatter(
                    y = health2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = health2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = health2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of Health to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of Health (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)
## Contribution Weight of Trust to Government

trust_to_gov2015 = df2015.trust_to_gov.tolist()
happiness_score2015 = df2015.happiness_score.tolist()
trust_to_gov2016 = df2016.trust_to_gov.tolist()
happiness_score2016 = df2016.happiness_score.tolist()
trust_to_gov2017 = df2017.trust_to_gov.tolist()
happiness_score2017 = df2017.happiness_score.tolist()

k = 20

for i in range(0,len(trust_to_gov2015)):
    trust_to_gov2015[i] = sum(trust_to_gov2015[i:i+k-1])/sum(df2015.happiness_score.values[i:i+k-1])*100
    happiness_score2015[i] = sum(happiness_score2015[i:i+k-1])/k
    trust_to_gov2016[i] = sum(trust_to_gov2016[i:i+k-1])/sum(df2016.happiness_score.values[i:i+k-1])*100
    happiness_score2016[i] = sum(happiness_score2016[i:i+k-1])/k
    trust_to_gov2017[i] = sum(trust_to_gov2017[i:i+k-1])/sum(df2017.happiness_score.values[i:i+k-1])*100
    happiness_score2017[i] = sum(happiness_score2017[i:i+k-1])/k

# Normalize happiness score by the maximum happiness score at the associated year    
happiness_score2015 = happiness_score2015/df2015.happiness_score[0]*100
happiness_score2016 = happiness_score2016/df2016.happiness_score[0]*100
happiness_score2017 = happiness_score2017/df2017.happiness_score[0]*100    
    
trace1 = go.Scatter(
                    y = trust_to_gov2015,
                    x = happiness_score2015,
                    mode = "lines",
                    name = "2015",
                   )

trace2 = go.Scatter(
                    y = trust_to_gov2016,
                    x = happiness_score2016,
                    mode = "lines",
                    name = "2016",
                   )

trace3 = go.Scatter(
                    y = trust_to_gov2017,
                    x = happiness_score2017,
                    mode = "lines",
                    name = "2017",
                   )

data = [trace1, trace2, trace3]
layout = dict(title = 'Contribution Weight of Trust to Government to Happiness',
              autosize=False,
              width=800,
              height=500,
              yaxis= dict(title= 'Contribution Weight of Trust to Government (%)',gridwidth=2, gridcolor='#bdbdbd'),
              xaxis= dict(title= 'Happiness Score (%)',gridwidth=2, gridcolor='#bdbdbd'),
              font=dict(size=14)
             )
fig = dict(data = data, layout = layout)
py.iplot(fig)