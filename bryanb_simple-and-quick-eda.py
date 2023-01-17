# Make                Car Make

# Model               Car Model

# Year                Car Year (Marketing)

# Engine Fuel Type    Engine Fuel Type

# Engine HP           Engine Horse Power (HP)

# Engine Cylinders    Engine Cylinders

# Transmission Type   Transmission Type

# Driven_Wheels       Driven Wheels

# Number of Doors     Number of Doors

# Market Category     Market Category

# Vehicle Size        Size of Vehicle

# Vehicle Style       Type of Vehicle

# highway MPG         Highway MPG

# city mpg            City MPG

# Popularity          Popularity (Twitter)

# MSRP                Manufacturer Suggested Retail Price
# Object manipulation

import statistics

import numpy as np

import pandas as pd

from collections import defaultdict



# Plot

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

import seaborn as sns
# Loading the data

df = pd.read_csv("../input/data.csv")

df.head(5)
print("The dataframe contains",df.shape[0],"rows and",df.shape[1],"columns\n")

print("The labels are",[df.columns[i] for i in range(df.shape[1])], "\n")



df.info()
# Construct data

index = df.groupby(['Year']).mean()['MSRP'].index.tolist()

mean_prices = df.groupby(['Year']).mean()['MSRP'].values.tolist()

std_prices = df.groupby(['Year']).std()['MSRP'].values.tolist()



# Data that will be used

price_per_year = pd.DataFrame(np.column_stack((mean_prices, std_prices)), columns=['Mean', 'Std'], index=index)



# Plot

fig = go.Figure()



fig.add_trace(go.Scatter(x=index, y=price_per_year.Mean,

                    mode='markers',

                    name='Mean Price'))



fig.add_trace(go.Scatter(x=index, y=price_per_year.Std,

                    mode='markers',

                    name='Std Price'))



fig.update_layout(title="Average MSRP per Year",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
np.argmax(df.loc[df['Year']==2008, 'MSRP'])

data_2008 = df[df['Year']==2008]

data_2008.iloc[326, :]
# Construct data

df_below_2000 = df[df['Year']<=2000]



# Plot

fig = px.box(df_below_2000, x="Year", y="MSRP")



reference_line = go.Scatter(x=[1989, 2001],

                            y=[10000, 10000],

                            mode="lines",

                            line=go.scatter.Line(color="red"),

                            showlegend=False)



fig.add_trace(reference_line)



fig.update_layout(title="Boxplots of MSRP per Year for cars sold before 2000",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
# Filtering

df_below_2000_filtered = df.loc[((df['Year']<=2000) & (df['MSRP']< 10000))]
# Plot

fig = px.box(df_below_2000_filtered, x="Year", y="MSRP")



fig.update_layout(title="Boxplots of MSRP per Year Filtered",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
# Filtering

df_after_2000 = df[df['Year']>2000]
fig = px.box(df_after_2000, x="Year", y="MSRP")



reference_line = go.Scatter(x=[2000, 2018],

                            y=[500000, 500000],

                            mode="lines",

                            line=go.scatter.Line(color="red"),

                            showlegend=False)



fig.add_trace(reference_line)



fig.update_layout(title="Boxplots of MSRP per Year",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
# Filtering

df_after_2000_filtered = df.loc[((df['Year']>2000) & (df['MSRP']< 500000))]
# Plot

fig = px.box(df_after_2000_filtered, x="Year", y="MSRP")



fig.update_layout(title="Boxplots of MSRP per Year Filtered",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
# Create index

dic = {1990+i : sum(df['Year']==1990+i) for i in range(28)}

x_dic = [1990 + i for i in range(28)]

y_dic = [dic[1990 + i] for i in range(28)]



# Plot

fig = go.Figure([go.Bar(x=x_dic, y=y_dic)])



fig.update_layout(title="Car year distribution",

                  xaxis_title="Year",

                  yaxis_title="Count Cars sold")





fig.show()
# Check Proportion of observations during last three years

print("Proportion of observations during last three years:",round(sum(y_dic[-3:])/sum(y_dic),2))
# Percentage of car per brand

counts = df['Make'].value_counts()*100/sum(df['Make'].value_counts())



# 10 most present labels

popular_labels = counts.index[:10]



# Plot

colors = ['lightslategray',] * len(popular_labels)

colors[0] = 'crimson'



fig = go.Figure(data=[go.Bar(

    x=counts[:10],

    y=popular_labels,

    marker_color=colors, # marker color can be a single color value or an iterable

    orientation='h'

)])



fig.update_layout(title_text='Proportion of Car brands in America (in %)',

                  xaxis_title="Percentage",

                  yaxis_title="Car Brand")
print(f"Over {len(counts)} different car brands, the 10 most recurrent car brands in that dataset represents {np.round(sum(counts[:10]))}% of the total number of cars !")
prices = df[['Make','MSRP']].loc[(df['Make'].isin(popular_labels))].groupby('Make').mean()

print(prices)
# Filtering

data_to_display = df[['Make','Year','MSRP']].loc[(df['Make'].isin(popular_labels)) & (df['Year'] > 2000)]



# Plot

fig = px.box(data_to_display, x="Year", y="MSRP")



fig.update_layout(title="MSRP over the 10 most represented Car brands",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
# Group categories (unleaded, flex-fuel, diesel, electric, natural gas)

df.loc[df['Engine Fuel Type']=='regular unleaded','Engine Fuel Type'] = 'unleaded'

df.loc[df['Engine Fuel Type']=='premium unleaded (required)','Engine Fuel Type'] = 'unleaded'

df.loc[df['Engine Fuel Type']=='premium unleaded (recommended)','Engine Fuel Type'] = 'unleaded'



df.loc[df['Engine Fuel Type']=='flex-fuel (unleaded/E85)','Engine Fuel Type'] = 'flex-fuel'

df.loc[df['Engine Fuel Type']=='flex-fuel (premium unleaded required/E85)','Engine Fuel Type'] = 'flex-fuel'

df.loc[df['Engine Fuel Type']=='flex-fuel (premium unleaded recommended/E85)','Engine Fuel Type'] = 'flex-fuel'

df.loc[df['Engine Fuel Type']=='flex-fuel (unleaded/natural gas)','Engine Fuel Type'] = 'flex-fuel'



eng = df.loc[~df['Year'].isin([2015,2016,2017]),'Engine Fuel Type'].value_counts()

eng2 = df.loc[df['Year'].isin([2015,2016,2017]),'Engine Fuel Type'].value_counts()



print('From last three years: \n')

print(eng, '\n')

print('From 1990 to 2014: \n')

print(eng2)



# Proportion before 2015

prop_eng_ft = pd.DataFrame({'Engine Fuel Type' : eng.index,

                            'Proportion': (eng/sum(eng)).tolist()})



# Proportion after 2015

prop_eng_ft2 = pd.DataFrame({'Engine Fuel Type' : eng2.index,

                            'Proportion 3years': (eng2/sum(eng2)).tolist()})
# Plot

fig = go.Figure()



fig.add_trace(go.Bar(

    x=prop_eng_ft['Engine Fuel Type'],

    y=prop_eng_ft['Proportion'],

    name='Proportion of cars per fuel type before 2015',

    marker_color='indianred'

))



fig.add_trace(go.Bar(

    x=prop_eng_ft2['Engine Fuel Type'],

    y=prop_eng_ft2['Proportion 3years'],

    name='Proportion of engine fuel type after 2015',

    marker_color='lightsalmon'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45,

                  title_text='Proportion of engine fuel type')



fig.show()
# Print correlation matrix

df.corr()
# Selecting only numerical features

list_numeric = list(df.describe().columns)
corr = df.corr()



# generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype = np.bool)



# return the indices for the upper triangle of an (n,m) array

mask[np.triu_indices_from(mask)] = True



# Plot

sns.set_style("white")

f, ax = plt.subplots(figsize=(11,7))

plt.title("Correlation matrix")

sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220,10, as_cmap=True),

            square=True, vmax = 1, center = 0, linewidths = .5, cbar_kws = {"shrink": .5})



plt.show()
# Plot

fig = px.histogram(df, x="Engine Cylinders", title='Engine cylinders',)

fig.show()
# Get index of highest number of cylinders

index = np.argmax(df['Engine Cylinders'])

df.loc[index,:]
print(np.argmax(df['MSRP']) == index)

print(np.argmax(df['MSRP']) == np.argmax(df['Engine HP']))
# Get data

data_pie = df['Transmission Type'].value_counts()



# Plot

fig = go.Figure(data=[go.Pie(labels=data_pie.index, values=data_pie.tolist(), textinfo='label+percent',

                             insidetextorientation='radial'

                            )])



fig.update_traces(hole=.3, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Pie chart of transmission type")



fig.show()
# Get data

data_pie = df['Driven_Wheels'].value_counts()



# Plot

fig = go.Figure(data=[go.Pie(labels=data_pie.index, values=data_pie.tolist(), textinfo='label+percent',

                             insidetextorientation='radial'

                            )])



fig.update_traces(hole=.3, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Pie chart of driven wheels variable")



fig.show()
# Create data

more_than_4_cyl = df[df['Engine Cylinders']>4]

less_than_4_cyl = df[~(df['Engine Cylinders']>4)]
# Plot

fig = go.Figure()



fig.add_trace(go.Bar(

    x=more_than_4_cyl['Number of Doors'].value_counts().index,

    y=(more_than_4_cyl['Number of Doors'].value_counts()/sum(more_than_4_cyl['Number of Doors'].value_counts())).tolist(),

    name='Number of doors for vehicles with more than 4 cylidners',

    marker_color='indianred'

))



fig.add_trace(go.Bar(

    x=less_than_4_cyl['Number of Doors'].value_counts().index,

    y=(less_than_4_cyl['Number of Doors'].value_counts()/sum(less_than_4_cyl['Number of Doors'].value_counts())).tolist(),

    name='Number of doors for vehicles with less than 4 cylidners',

    marker_color='lightsalmon'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45,

                  title_text='Proportion of Number of Doors')

fig.show()
# Get data

data_pie = df['Vehicle Size'].value_counts()



# Plot

fig = go.Figure(data=[go.Pie(labels=data_pie.index, values=data_pie.tolist(), textinfo='label+percent',

                             insidetextorientation='radial'

                            )])



fig.update_traces(hole=.3, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Pie chart of driven wheels variable")



fig.show()
# Plot

fig = go.Figure()



fig.add_trace(go.Bar(

    x=df['Make'].unique(),

    y=df.groupby(['Make']).mean()['Popularity'],

    name='Proportion of cars per fuel type before 2015',

    marker_color='indianred'

))



fig.update_layout(xaxis_tickangle=-45,

                  title_text='Popularity of different car brands',

                  xaxis_title="Car Brand",

                  yaxis_title="Popularity")

fig.show()
# Get data

data = np.log10(df.MSRP)



# Plot

fig = ff.create_distplot([data], ['G1'], bin_size=.05,

                         curve_type='normal', # override default 'kde'

                         show_rug=False,

                         show_curve=False,

                         histnorm='probability')



fig.update_layout(title_text='MSRP Distribution',

                  xaxis_title="Log10 MSRP")



fig.show()
# Some observations

print(sum(df['MSRP']>400000),"cars worth more than 400k$")

print(sum(df['MSRP']>200000),"cars worth more than 200k$")

print(sum(df['MSRP']>50000),"cars worth more than 50k$")

print("Most expensive car costs", max(df['MSRP']))

print("There are",sum(df['MSRP']<5000),"cars worth less than 5k$")
# Filtering

below_5k_per_year_mean = df.loc[df['MSRP']<5000].groupby(['Year']).mean()['MSRP']

below_5k_per_year_count = df.loc[df['MSRP']<5000].groupby(['Year']).count()['MSRP']
# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(

    go.Scatter(x=below_5k_per_year_mean.index, 

               y=below_5k_per_year_mean, 

               name="Mean MSRP per year",

               marker_color='indianred'),

    secondary_y=True,

)



fig.add_trace(go.Bar(

    x=below_5k_per_year_count.index,

    y=below_5k_per_year_count,

    name='Count cars per year',

    marker_color='lightsalmon'

))



# Set x-axis title

fig.update_xaxes(title_text="Year")



# Set y-axes titles

fig.update_yaxes(title_text="<b>Count</b> cars", secondary_y=False)

fig.update_yaxes(title_text="<b>Mean MSRP</b> (in USD)", secondary_y=True)



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', 

                  xaxis_tickangle=-45,

                  title_text='Age of less than 5k USD sold cars')



fig.show()
# Copy of the dataset

df_to_clean = df.copy()

print(df_to_clean.describe().columns)
# Check NAs per column

df_to_clean.isnull().sum()
index = df_to_clean.isnull().sum()>0

print("The feature which contain missing values are\n",[df.columns[index][i] for i in range(sum(index))])
# Drop rows containing at least one NA

df_drop_rows_with_na = df_to_clean.dropna()
(df.shape[0] - df_drop_rows_with_na.shape[0]) / df.shape[0]
print(df_to_clean['Engine Fuel Type'].value_counts())

print(sum(df_to_clean['Engine Fuel Type'].value_counts()))
print(df_to_clean['Engine Fuel Type'].unique())

engine_fuel_type_mapping = {label:idx for idx,label in enumerate(df_to_clean['Engine Fuel Type'].unique())}

print("\n",engine_fuel_type_mapping)



# Apply once

df_to_clean['Engine Fuel Type'] = df_to_clean['Engine Fuel Type'].map(engine_fuel_type_mapping)
indexes_eft = np.where(df_to_clean.loc[:,'Engine Fuel Type']==5)

df_to_clean.loc[indexes_eft[0][0]:(indexes_eft[0][0]+5),:]
# Inputing

df_to_clean.loc[indexes_eft[0],'Engine Fuel Type'] = 1
# Check the 10 category which corresponds to NA has been removed

print(df_to_clean['Engine Fuel Type'].unique())
# Plot

fig = px.box(df, y="Engine HP")



fig.update_layout(title="MSRP over the 10 most represented Car brands",

                  xaxis_title="Year",

                  yaxis_title="MSRP")



fig.show()
# Index of each NA values in 'Engine HP'

indexes_engine_hp = np.where(df_to_clean['Engine HP'].isnull())

print("Indexes of each NA in 'Engine HP'",indexes_engine_hp[0])



# Numbers of NA (does it fit with the dataframe ?)

print(len(indexes_engine_hp[0]))



# Mean inputation

for i in range(len(indexes_engine_hp[0])):

    df_to_clean.loc[indexes_engine_hp[0][i],'Engine HP'] = np.mean(df_to_clean['Engine HP'])
print(df_to_clean['Engine Cylinders'].value_counts(), "\n")

print(sum(df_to_clean['Engine Cylinders'].value_counts()))
indexes_engine_cyl = np.where(df_to_clean['Engine Cylinders'].isnull())

print('There are',len(indexes_engine_cyl[0]),'missing values as expected')
display_cars = pd.DataFrame(df_to_clean.loc[indexes_engine_cyl[0],['Make','Model']].values, columns = ['Make','Model'])

print(display_cars.head(30))
# Solution to reduce redundancy

list(display_cars.groupby(['Make','Model']).groups.keys())
df_to_clean.loc[indexes_engine_cyl[0][0:10],'Engine Cylinders'] = 0 

df_to_clean.loc[indexes_engine_cyl[0][10:],'Engine Cylinders'] = 0
indexes_nb_doors = np.where(df_to_clean['Number of Doors'].isnull())

df_to_clean.loc[indexes_nb_doors]
df_to_clean.loc[indexes_nb_doors[0][0],'Number of Doors'] = 3

df_to_clean.loc[indexes_nb_doors[0][1:],'Number of Doors'] = 5
df_to_clean.loc[np.where(df_to_clean['Market Category'].isnull())[0],'Market Category'] = 'No category'



# 5 most frequent categories

df_to_clean['Market Category'].value_counts().head(5)
# Save dataframe

df_to_clean.to_csv('cars_cleaned.csv')