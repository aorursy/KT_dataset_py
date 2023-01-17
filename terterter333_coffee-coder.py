## Import 

import numpy as np    

import pandas as pd   

import matplotlib.pyplot as plt

import plotly.graph_objects as go  

from scipy.stats.stats import pearsonr



## Plotly

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go  

from plotly.offline import iplot
## I'm not sure about description of CodingWithoutCoffee cols. If the label is 'Yes', is that mean a person "can" do coding without drinking coffee ?

df = pd.read_csv('../input/coffee-and-code/CoffeeAndCodeLT2018.csv')

print(f"Shape: {df.shape}")



## Preview 5 first rows of the dataset 

df.head()
#Basic inference statistics

df.describe()
## The dataset contain small of Nan values. CoffeeType and AgeRangeis are missing 1% and 2% respectively

## Assume that there is no duplicate row

df.isna().mean()
#Inspect outlier for CodingHours

trace0 = go.Box(

    y=df[df.Gender == 'Male'].CodingHours,

    name = 'Male',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=df[df.Gender == 'Female'].CodingHours,

    name = 'Female',

    marker = dict(

        color = 'rgb(255, 113, 181)',

    )

)



data = [trace0, trace1]

layout = dict(title = 'Boxplot of CodingHours',

              xaxis= dict(title= 'Gender',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'CodingHours',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig)

#Inspect outlier for CoffeeCupsPerDay

trace0 = go.Box(

    y=df[df.Gender == 'Male'].CoffeeCupsPerDay,

    name = 'Male',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=df[df.Gender == 'Female'].CoffeeCupsPerDay,

    name = 'Female',

    marker = dict(

        color = 'rgb(255, 113, 181)',

    )

)



data = [trace0, trace1]

layout = dict(title = 'Boxplot of CodingHours',

              xaxis= dict(title= 'Gender',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'CoffeeCupsPerDay',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig)

# High level    

print(f"Overall average consumption for coder is {round(df.CoffeeCupsPerDay.mean(),2)} cups per day")

print(f"Average consumption for male coder is {round(df[df.Gender == 'Male'].CoffeeCupsPerDay.mean(),2)} cups per day")

print(f"Average consumption for female coder is {round(df[df.Gender == 'Female'].CoffeeCupsPerDay.mean(),2)} cups per day")
mean_groupby_df = df.groupby(['AgeRange','Gender']).mean().reset_index()  

mean_groupby_df.head()
# At lower level

fig = go.Figure()

fig.add_trace(go.Bar(x=mean_groupby_df[mean_groupby_df.Gender == 'Male'].CoffeeCupsPerDay,

                    y=mean_groupby_df[mean_groupby_df.Gender == 'Male'].AgeRange,

                    name='Male',

                    marker_color='#4682B4',

                    orientation='h'

                    ))

fig.add_trace(go.Bar(x=mean_groupby_df[mean_groupby_df.Gender == 'Female'].CoffeeCupsPerDay,

                    y=mean_groupby_df[mean_groupby_df.Gender == 'Female'].AgeRange,

                    name='Female',

                    marker_color='#FFB6C1',

                    orientation='h'

                    ))



fig.update_layout(

    title='Average Coffee Consumption groupby AgeRange and Gender',



    xaxis=dict(

        title='Average cups per day',

        titlefont_size=16,

        tickfont_size=14,),

    yaxis=dict(

        title='Age range',

        titlefont_size=16,

        tickfont_size=14,),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
# Outlier row

df[(df.Gender == 'Female') & (df.AgeRange == '40 to 49')]  
# Measures linear correlation between two variables    

print(f'Pearson correlation btw CodingHours and CoffeeCupsPerDay is {round(pearsonr(np.array(df.CodingHours),np.array(df.CoffeeCupsPerDay))[0],3)}')
fig = px.scatter(df.dropna(), x="CoffeeCupsPerDay", y="CodingHours", facet_col="Gender", trendline="ols",opacity=0.25)

fig.update_layout(

    title_text='Scatter plot between CodingHours and CoffeeCupsPerDay', 



    yaxis=dict(

        title='Coding hours',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
fig = px.scatter(df.dropna(), x="CoffeeCupsPerDay", y="CodingHours", facet_col='AgeRange',facet_col_spacing=0.06,color='Gender', trendline="ols",opacity=0.25)

fig.update_layout(

    title_text='Scatter plot between CodingHours and CoffeeCupsPerDay', 

    yaxis=dict(

        title='Coding hours',

        titlefont_size=16,

        tickfont_size=14)

)

fig.show()
fig = px.scatter_3d(df, x=df.CodingHours, y=df.CoffeeCupsPerDay, z=df.AgeRange,

              color=df.Gender,opacity=0.5)

fig.update_layout(

    title_text='3D-Scatter plot of CodingHours, CoffeeCupsPerDay and AgeRange')

fig.show()
# High level

df.groupby(['CoffeeType']).agg(['mean', 'count']).CodingHours
# 2 cols groupby

mean_groupby_df = df.groupby(['CoffeeType','Gender']).agg(['mean', 'count']).CodingHours.reset_index()  

mean_groupby_df.head()
# Low level       

mean_groupby_df = df.groupby(['CoffeeType','Gender']).agg(['mean', 'count']).CodingHours.reset_index()  

fig = go.Figure()

fig.add_trace(go.Bar(x=mean_groupby_df[mean_groupby_df.Gender == 'Male']['mean'],

                    y=mean_groupby_df[mean_groupby_df.Gender == 'Male'].CoffeeType,

                    name='Male',

                    marker_color='#4682B4',

                    orientation='h'

                    ))

fig.add_trace(go.Bar(x=mean_groupby_df[mean_groupby_df.Gender == 'Female']['mean'],

                    y=mean_groupby_df[mean_groupby_df.Gender == 'Female'].CoffeeType,

                    name='Female',

                    marker_color='#FFB6C1',

                    orientation='h'

                    ))



fig.update_layout(

    title='Average Coffee Hours groupby Types of coffee and Gender',



    xaxis=dict(

        title='Mean of coding hours',

        titlefont_size=16,

        tickfont_size=14,),

    yaxis=dict(

        title='Coffee Type',

        titlefont_size=16,

        tickfont_size=14,),

    barmode='group',  

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
# 3 cols Groupby 

three_groupby_df = df.groupby(['AgeRange','CoffeeType','Gender']).agg(['mean','count']).reset_index()

three_groupby_df.head()
fig = make_subplots(rows=1, cols=2,x_title='AgeRange',horizontal_spacing=0.32,column_titles=['Male Coder','Female Coder'])

fig.add_trace((go.Heatmap(

    x=three_groupby_df[three_groupby_df.Gender == 'Male'].AgeRange,

    y=three_groupby_df[three_groupby_df.Gender == 'Male'].CoffeeType,

    z=three_groupby_df[three_groupby_df.Gender == 'Male'].CodingHours['mean'],

    name = 'Male',

    zmin=three_groupby_df.CodingHours['mean'].min(), 

    zmax=three_groupby_df.CodingHours['mean'].max(),

    hovertemplate='Age: %{x}<br>Types: %{y}<br>CodingHour: %{z}<extra></extra>'

    )),1, 1)

fig.add_trace((go.Heatmap(

    x=three_groupby_df[three_groupby_df.Gender == 'Female'].AgeRange,

    y=three_groupby_df[three_groupby_df.Gender == 'Female'].CoffeeType,

    z=three_groupby_df[three_groupby_df.Gender == 'Female'].CodingHours['mean'],

    name = 'Female',

    zmin=three_groupby_df.CodingHours['mean'].min(), 

    zmax=three_groupby_df.CodingHours['mean'].max(),

    hovertemplate='Age: %{x}<br>Types: %{y}<br>CodingHour: %{z}<extra></extra>'

    )),1, 2)



fig.update_traces(hoverinfo="all", colorbar=dict(title='CodingHourRange'))



fig.update_layout(

    title_text="Heat map of Average CodingHours groupby Age, Type and Gender")

fig.show()       
df.CoffeeSolveBugs.value_counts(normalize=True)               
fig = px.histogram(df.dropna(), x="CoffeeSolveBugs",facet_col="AgeRange",facet_col_spacing=0.06,

                   category_orders={"AgeRange": ['Under 18','18 to 29', '30 to 39', '40 to 49', '50 to 59']},histnorm='percent')



fig.update_layout(

    title_text='Histogram of CoffeeSolveBugs Groupby AgeRange', 



    yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()  
# There is only one data point for coder (50 to 59)

df[df.AgeRange == '50 to 59']
fig = go.Figure()

fig.add_trace(go.Histogram(

    x=df[df.Gender == 'Male'].CoffeeSolveBugs,

    histnorm='percent',

    name='Male',

    opacity=0.9

))

fig.add_trace(go.Histogram(

    x=df[df.Gender == 'Female'].CoffeeSolveBugs,

    histnorm='percent',

    name='Female', 

    opacity=0.9

))



fig.update_layout(

    title_text='Histogram of CoffeeSolveBugs Groupby Gender', # title of plot

    xaxis_title_text='Coffee Solve Bugs', # xaxis label

    yaxis_title_text='Percent', # yaxis label

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show()  
# Create column SolveBugs value  

# Yes: 1

# Sometimes: 0.5

# No: 0

df['SolveBugs_val'] = df.CoffeeSolveBugs.apply(lambda x: 1 if x == 'Yes' else (0.5 if x == 'Sometimes' else 0))



# 2 cols Groupby

groupby_df = df.groupby(['AgeRange','CoffeeType']).mean().reset_index()



fig = go.Figure(go.Heatmap(

    x=groupby_df.AgeRange,

    y=groupby_df.CoffeeType,

    z=groupby_df.SolveBugs_val,

    hovertemplate='Age: %{x}<br>Types: %{y}<br>Bug Solving Value: %{z}<extra></extra>',

    colorbar=dict(title='BugSolvingValueRange')

    

))

fig.update_layout(

    title_text='Heatmap of Bug solving value groupby Age and Coffee type', 

    xaxis_title_text='AgeRange', 

    yaxis_title_text='CoffeeType', 

)



fig.show()

  
# 3 cols Groupby 

three_groupby_df = df.groupby(['AgeRange','CoffeeType','Gender']).agg(['mean','count']).reset_index()

three_groupby_df.head()
# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2,x_title='AgeRange',horizontal_spacing=0.32,column_titles=['Male Coder','Female Coder'])

fig.add_trace((go.Heatmap(

    x=three_groupby_df[three_groupby_df.Gender == 'Male'].AgeRange,

    y=three_groupby_df[three_groupby_df.Gender == 'Male'].CoffeeType,

    z=three_groupby_df[three_groupby_df.Gender == 'Male'].SolveBugs_val['mean'],

    name = 'Male',

    zmin=three_groupby_df.SolveBugs_val['mean'].min(), 

    zmax=three_groupby_df.SolveBugs_val['mean'].max(),

    hovertemplate='Age: %{x}<br>Types: %{y}<br> Bug Solving Value: %{z}<extra></extra>'

    )),1, 1)

fig.add_trace((go.Heatmap(

    x=three_groupby_df[three_groupby_df.Gender == 'Female'].AgeRange,

    y=three_groupby_df[three_groupby_df.Gender == 'Female'].CoffeeType,

    z=three_groupby_df[three_groupby_df.Gender == 'Female'].SolveBugs_val['mean'],

    name = 'Female',

    zmin=three_groupby_df.SolveBugs_val['mean'].min(), 

    zmax=three_groupby_df.SolveBugs_val['mean'].max(),

    hovertemplate='Age: %{x}<br>Types: %{y}<br>Bug Solving Value: %{z}<extra></extra>'

    

    )),1, 2)



fig.update_traces(hoverinfo="all",colorbar=dict(title='BugSolvingValueRange'))



fig.update_layout(

    title_text="Heat map of Bug solving value groupby Age, Type and Gender")

fig.show()  
fig = go.Figure()      

fig.add_trace(go.Histogram(

    x=df.CoffeeType,

    histnorm='percent',

    marker_color='#4682B4',

    opacity=0.9

))

fig.update_layout(

    title_text='Histogram of Coffee preference', # title of plot

    xaxis_title_text='Types of coffee', # xaxis label

    yaxis_title_text='Percent', # yaxis label

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars of the same location coordinates

)     

fig.show()
# fig = go.Figure()  

# fig.add_trace(go.Histogram(

#     x=df[df.Gender == 'Male'].CoffeeType,

#     histnorm='percent',

#     name='Male', # name used in legend and hover labels

#     marker_color='#4682B4',

#     opacity=0.9

# ))

# fig.add_trace(go.Histogram(

#     x=df[df.Gender == 'Female'].CoffeeType,

#     histnorm='percent',

#     name='Female',

#     marker_color='#FFB6C1',

#     opacity=0.95

# ))



# fig.update_layout(

#     title_text='Histogram of Coffee preference Groupby Gender', # title of plot

#     xaxis_title_text='Types of coffee', # xaxis label

#     yaxis_title_text='Percent', # yaxis label

#     bargap=0.2, # gap between bars of adjacent location coordinates

#     bargroupgap=0.1 # gap between bars of the same location coordinates

# )



# fig.show()



fig = px.histogram(df.dropna(), x="CoffeeType",facet_col="Gender", histnorm='percent')



fig.update_layout(

    title_text='Histogram of Coffee preference Groupby Gender', 



    yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
fig = px.histogram(df.dropna(), x="CoffeeType",facet_col="AgeRange",facet_col_spacing=0.03,

                   category_orders={"AgeRange": ['Under 18','18 to 29', '30 to 39', '40 to 49', '50 to 59']},histnorm='percent')



fig.update_layout(

    title_text='Histogram of Coffee preference Groupby AgeRange', 



    yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,))

fig.show()
fig = px.histogram(df.dropna(), x="CoffeeType",facet_row='Gender',facet_col="AgeRange",facet_col_spacing=0.03

                   ,category_orders={"AgeRange": ['Under 18','18 to 29', '30 to 39', '40 to 49', '50 to 59']}, histnorm='percent')



fig.update_layout(

    title_text='Histogram of Coffee preference Groupby AgeRange', 



    yaxis=dict(

        title='Percent',

        titlefont_size=16,

        tickfont_size=14,)

)

fig.show()
fig = px.pie(df, names='CoffeeTime', hole=.35)

fig.update_layout(

    title_text="Pie chart of CoffeeTime (Overall)")

fig.show()
male_ct = df[df.Gender == 'Male'].CoffeeTime.value_counts(normalize=True)

female_ct = df[df.Gender == 'Female'].CoffeeTime.value_counts(normalize=True)

# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=male_ct.index, values=male_ct.values, name="Male"),1, 1)

fig.add_trace(go.Pie(labels=female_ct.index, values=female_ct.values, name="Female"),1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Pie chart of CoffeeTime For each Gender",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Male', x=0.185, y=0.5, font_size=16, showarrow=False),

                 dict(text='Female', x=0.835, y=0.5, font_size=16, showarrow=False)])

fig.show()  
fig = go.Figure(go.Histogram2d(

        x=df.CoffeeTime,

        y=df.CoffeeType,

        hovertemplate='CoffeeTime: %{x}<br>Types: %{y}<br>Frequency: %{z}<extra></extra>',

        colorbar=dict(title='Frequency')

    ))

fig.update_layout(

    title_text='Density plot of Types of coffee & Coffee time', 

    xaxis_title_text='Coffee time', 

    yaxis_title_text='Types of coffee')

fig.show()