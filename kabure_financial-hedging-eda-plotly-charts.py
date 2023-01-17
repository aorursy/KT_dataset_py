#Load the librarys

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams



import plotly.tools as tls

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import warnings



from scipy import stats



%matplotlib inline



# figure size in inches

rcParams['figure.figsize'] = 12,6
df_features = pd.read_csv("../input/real_estate_db.csv", encoding='ISO-8859-1' )



del df_features['BLOCKID']

del df_features['UID'] 
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
#looking the shape of data

resumetable(df_features)[:43]
#looking the shape of data

resumetable(df_features)[43:]
#Looking the data

df_features.head()
percentual_types = round(df_features["type"].value_counts(), 2)



types = round(df_features["type"].value_counts() / len(df_features["type"]) * 100,2)



labels = list(types.index)

values = list(types.values)



trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']), text = percentual_types.values,)



layout = go.Layout(title='Distribuition of Types', legend=dict(orientation="h"));



fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
state_count = df_features["state"].value_counts()

city_count = df_features.city.value_counts()

place_count = df_features.place.value_counts()

primary_count = df_features.primary.value_counts()
trace1 = go.Bar(x=state_count[:20].values[::-1],

                y=state_count[:20].index[::-1],

                orientation='h', visible=True,

                      name='Top 20 States',

                      marker=dict(

                          color=city_count[:20].values[::-1],

                          colorscale = 'Viridis',

                          reversescale = True

                      ))



trace2 = go.Bar(x=city_count[:20].values[::-1],

                      y=city_count[:20].index[::-1],

                      orientation = 'h', visible=False, 

                      name='TOP 20 Citys',

                      marker=dict(

                          color=city_count[:20].values[::-1],

                          colorscale = 'Viridis',

                          reversescale = True

                      ))



trace3 = go.Histogram(y=sorted(df_features['type'], reverse=True), histnorm='percent', orientation='h', visible=False, 

                      name='Type Count')



trace4 = go.Bar(x=place_count[:20].values[::-1],

                y=place_count[:20].index[::-1],

                orientation='h', visible=False, 

                name='Top 20 Place',

                marker=dict(

                    color=city_count[:20].values[::-1],

                    colorscale = 'Viridis',

                    reversescale = True

                      ))



data = [trace1, trace2, trace3, trace4]



updatemenus = list([

    dict(active=-1,

         x=-0.15,

         buttons=list([  

            dict(

                label = 'State Count',

                 method = 'update',

                 args = [{'visible': [True, False, False, False]}, 

                         {'title': 'TOP 20 State Count'}]),

             

             dict(

                  label = 'City Count',

                 visible=True,

                 method = 'update',

                 args = [{'visible': [False, True, False, False]},

                     {'title': 'TOP 20 City Count'}]),



            dict(

                 label = 'Type Count',

                 method = 'update',

                 args = [{'visible': [False, False, True, False]},

                     {'title': 'Type Counts'}]),



            dict(

                 label = 'Place Count',

                 method = 'update',

                 args = [{'visible': [False, False, False, True]},

                     {'title': ' Top 20 Place Count'}])

        ]),

    )

])





layout = dict(title='The count of the principal Categorical Features <br>(Select from Dropdown)', 

              showlegend=False,

              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)



iplot(fig)

df_features['ALand_div_1M'] = np.log(df_features['ALand'] / 1000000)


trace1  = go.Box(

    x=df_features[df_features.city.isin(city_count[:15].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:15].index.values)]['rent_median'], 

    showlegend=False, visible=True

)

                        

trace2  = go.Box(

    x=df_features[df_features.city.isin(city_count[:15].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:15].index.values)]['family_median'], 

    showlegend=False, visible=False

)

                

trace3 = go.Box(

    x=df_features[df_features.city.isin(city_count[:15].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:15].index.values)]['hi_median'],

    showlegend=False, visible=False

)



trace4 = go.Box(

    x=df_features[df_features.city.isin(city_count[:15].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:15].index.values)]['hc_mortgage_mean'],

    showlegend=False, visible=False

)



data = [trace1, trace2, trace3, trace4]



updatemenus = list([

    dict(active=-1,

         x=-0.15,

         buttons=list([  

             

            dict(

                label = 'City Rent Boxplot',

                 method = 'update',

                 args = [{'visible': [True, False, False, False]}, 

                     {'title': 'TOP 15 Citys - Rent Median'}]),

             

             dict(

                  label = 'City Family Boxplot',

                 method = 'update',

                 args = [{'visible': [False, True, False, False]},

                     {'title': 'TOP 15 Citys - Family Income Median'}]),



            dict(

                 label = 'City House Inc',

                 method = 'update',

                 args = [{'visible': [False, False, True, False]},

                     {'title': 'TOP 15 Citys - House income Median'}]),



            dict(

                 label =  'City HC Mortage',

                 method = 'update',

                 args = [{'visible': [False, False, False, True]},

                     {'title': 'TOP 15 Citys - Home Cost Mortage'}])

        ]),

    )

])



layout = dict(title='Citys BoxPlots of Medians <br>(Select metrics from Dropdown)', 

              showlegend=False,

              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)



iplot(fig, filename='dropdown')
city_count = df_features.city.value_counts()



#First plot

trace0 = go.Box(

    x=df_features[df_features.city.isin(city_count[:10].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:10].index.values)]['rent_median'], 

    showlegend=False

)



#Second plot

trace1 = go.Box(

    x=df_features[df_features.city.isin(city_count[:10].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:10].index.values)]['family_median'], 

    showlegend=False

)



#Second plot

trace2 = go.Box(

    x=df_features[df_features.city.isin(city_count[:10].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:10].index.values)]['hc_mortgage_median'], 

    showlegend=False

)



#Third plot

trace3 = go.Histogram(

    x=df_features[df_features.city.isin(city_count[:20].index.values)]['city'], histnorm='percent',

    showlegend=False

)

#Third plot

trace4 = go.Histogram(

    x=np.log(df_features['family_median']).sample(5000), histnorm='percent', autobinx=True,

    showlegend=True, name='Family'

)



#Third plot

trace5 = go.Histogram(

    x=np.log(df_features['hc_mortgage_median']).sample(5000), histnorm='percent', autobinx=True,

    showlegend=True, name='HC mort'

)



#Third plot

trace6 = go.Histogram(

    x=np.log(df_features['rent_median']).sample(5000), histnorm='percent', autobinx=True,

    showlegend=True, name="Rent"

)



#Creating the grid

fig = tls.make_subplots(rows=2, cols=3, specs=[[{'colspan': 2}, None, {}], [{}, {}, {}]],

                          subplot_titles=("Citys Count",

                                          "Medians Distribuition", 

                                          "HC Morttage Median",

                                          "Family Median", 

                                          "Rent Median"))



#setting the figs

fig.append_trace(trace0, 2, 1)

fig.append_trace(trace1, 2, 3)

fig.append_trace(trace2, 2, 2)

fig.append_trace(trace3, 1, 1)

fig.append_trace(trace4, 1, 3)

fig.append_trace(trace5, 1, 3)

fig.append_trace(trace6, 1, 3)



fig['layout'].update(showlegend=True, title="Some Top Citys Distribuitions")



iplot(fig)


#First plot

trace0 = go.Box(

    x=df_features[df_features.city.isin(city_count[:10].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:10].index.values)]['rent_median'], 

    showlegend=False

)



#Second plot

trace1 = go.Box(

    x=df_features[df_features.city.isin(city_count[:10].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:10].index.values)]['family_median'], 

    showlegend=False

)



#Second plot

trace2 = go.Box(

    x=df_features[df_features.city.isin(city_count[:10].index.values)]['city'],

    y=df_features[df_features.city.isin(city_count[:10].index.values)]['hc_mortgage_median'], 

    showlegend=False

)



#Third plot

trace3 = go.Histogram(

    x=df_features[df_features.city.isin(city_count[:20].index.values)]['city'], histnorm='percent',

    showlegend=False

)



#Creating the grid

fig = tls.make_subplots(rows=2, cols=3, specs=[[{'colspan': 3}, None, None], [{}, {}, {}]],

                          subplot_titles=("City Count",

                                          "Rent Median by City",

                                          "HC Morttage Median by City",

                                          "Family Median by City"

                                          ))

#setting the figs

fig.append_trace(trace0, 2, 1)

fig.append_trace(trace1, 2, 3)

fig.append_trace(trace2, 2, 2)

fig.append_trace(trace3, 1, 1)



fig['layout'].update(showlegend=True, title="Some City Distribuitions")

iplot(fig)


#First plot

trace0 = go.Box(

    x=df_features[df_features.state.isin(state_count[:10].index.values)]['state'],

    y=df_features[df_features.state.isin(state_count[:10].index.values)]['hs_degree'],

    name="Top 10 States", showlegend=False

)



#Second plot

trace1 = go.Box(

    x=df_features[df_features.state.isin(state_count[:10].index.values)]['state'],

    y=df_features[df_features.state.isin(state_count[:10].index.values)]['family_median'],

    name="Top 15 Sucessful", showlegend=False

)



#Third plot

trace2 = go.Histogram(

    x=df_features[df_features.place.isin(place_count[:20].index.values)]['place'],

    histnorm='percent', name="Top 20 Place's", showlegend=False             

)



#Creating the grid

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('HS Degree Median TOP 10 States',

                                          'Family Median TOP 10 States', 

                                          "Top 20 Most Frequent Places"))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)



fig['layout'].update(showlegend=True, title="Top Frequency States")



iplot(fig)
#First plot

trace0 = go.Box(

    x=df_features['type'],

    y=df_features['rent_median'], 

    showlegend=False

)



#Second plot

trace1 = go.Box(

    x=df_features['type'],

    y=df_features['family_median'], 

    showlegend=False

)



#Second plot

trace2 = go.Histogram(

    x=df_features['type'], histnorm="percent", 

    showlegend=False

)



trace3 = go.Scatter(

    x=df_features['rent_median'], 

    y=df_features['family_median'],

    showlegend=False,

    mode = 'markers'

)



#Creating the grid

fig = tls.make_subplots(rows=2, cols=3, specs=[[{}, {}, {}], [{'colspan': 3}, None, None]],

                          subplot_titles=("Rent Median by Type",

                                          "Type Count",

                                          "Family Median by Type", 

                                          "Rent Median x Family Median"))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 3)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 2, 1)



fig['layout'].update(showlegend=True, 

                     title="Some Type Distribuitions")



iplot(fig)


#First plot

trace0 = go.Box(

    x=df_features[df_features.state.isin(state_count[:10].index.values)]['state'],

    y=df_features[df_features.state.isin(state_count[:10].index.values)]['rent_median'],

    name="Top 10 States", showlegend=False

)



#Second plot

trace1 = go.Box(

    x=df_features[df_features.state.isin(state_count[:10].index.values)]['state'],

    y=df_features[df_features.state.isin(state_count[:10].index.values)]['hc_mortgage_median'],

    name="Top 15 Sucessful", showlegend=False

)



#Third plot

trace2 = go.Histogram(

    x=df_features[df_features.state.isin(state_count[:20].index.values)]['state'],

    histnorm='percent', name="Top 20 States's", showlegend=False             

)



#Creating the grid

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Rent Median TOP 10 States',

                                          'Mortage Median TOP 10 States', 

                                          "Top 20 Most Frequent States"))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)



fig['layout'].update(showlegend=True, title="Top Frequency States")



iplot(fig)
cat_feat = df_features.loc[:, df_features.dtypes == object].columns

num_feat = df_features.loc[:, df_features.dtypes != object].columns
female_male  = ['hs_degree', 'hs_degree_male', 'hs_degree_female', 'male_age_mean',

                'male_age_median', 'male_age_stdev', 'male_age_sample_weight',

                'male_age_samples', 'female_age_mean', 'female_age_median',

                'female_age_stdev', 'female_age_sample_weight', 'female_age_samples']