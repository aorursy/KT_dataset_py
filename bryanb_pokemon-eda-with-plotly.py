import os

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from plotly.subplots import make_subplots

import plotly.express as px

import plotly.graph_objects as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



init_notebook_mode()
filepath = '../input/'

pokemon = pd.read_csv(filepath + 'complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv').drop('Unnamed: 0', axis = 1)



columns_to_drop = ['japanese_name', 'german_name', 'against_normal', 'against_fire',

                  'against_water', 'against_electric', 'against_grass', 'against_ice',

                  'against_fight', 'against_poison', 'against_ground', 'against_flying',

                  'against_psychic', 'against_bug', 'against_rock', 'against_ghost',

                  'against_dragon', 'against_dark', 'against_steel', 'against_fairy']



pokemon = pokemon.drop(columns_to_drop, axis = 1)
pokemon.info()
# Select mega pokemons, dinamax and alolan pokemons

mega_pokemons = pokemon.index[pokemon['name'].apply(lambda x: 'Mega ' in x)].tolist()

dinamax_pokemons = pokemon.index[pokemon['name'].apply(lambda x: 'max' in x)].tolist()

alolan_pokemons = pokemon[pokemon.name.apply(lambda x: 'Alolan' in x) == True].index.tolist()



# Concatenate

to_delete = np.concatenate((mega_pokemons, dinamax_pokemons, alolan_pokemons))



# Remove

pokemon = pokemon.drop(to_delete, axis=0)
# Check NAs

pokemon.isnull().sum()
# Clear cache

del(mega_pokemons, dinamax_pokemons, alolan_pokemons, to_delete)
fig = px.histogram(pokemon, x="total_points",

                   marginal="box",

                   hover_data=pokemon.columns)



fig.update_layout(

    title="Total points distribution")



fig.show()
# Get index and print row of pokemon having highest total_points

highest_tot_points_idx = pokemon['total_points'].idxmax()

pokemon.loc[highest_tot_points_idx,:]
def find_min_and_max(column_name):

    '''

    Get pokemon name according to its max and min attribute: column_name

    column_name: array of int or float

    '''

    

    # Find max

    max_index = pokemon[column_name].idxmax()

    max_pokemon = pokemon.loc[max_index, 'name']

    

    # Find min

    min_index = pokemon[column_name].idxmin()

    min_pokemon = pokemon.loc[min_index, 'name']

    

    print(f'Pokemon with min {column_name}: {min_pokemon}\nPokemon with max {column_name}: {max_pokemon}\n')

    return max_index, min_index
# Create dict for min and max values of selected columns

columns = ['attack', 'defense', 'sp_attack', 'sp_defense', 'hp', 'speed', 'catch_rate']

min_dict = {}

max_dict = {}

min_pok = {}

max_pok = {}



for colm in columns:

    max_index, min_index = find_min_and_max(colm)

    max_dict[colm] = pokemon.loc[max_index, colm]

    min_dict[colm] = pokemon.loc[min_index, colm]

    max_pok[colm] = pokemon.loc[max_index, 'name']

    min_pok[colm] = pokemon.loc[min_index, 'name']
fig = go.Figure([go.Bar(x=columns, 

                        y=list(max_dict.values()), 

                        hovertext=[f"{columns[i]}, {list(max_dict.values())[i]}, {list(max_pok.values())[i]}" for i in range(len(columns)) ], 

                        name="Highest")])



fig.add_trace(go.Bar(x=columns, 

                     y=list(min_dict.values()), 

                     hovertext=[f"{columns[i]}, {list(min_dict.values())[i]}, {list(min_pok.values())[i]}" for i in range(len(columns)) ], 

                     name='Lowest'))



fig.update_layout(

    title="Highest vs Lowest barplot")



fig.show()
def are_row_wise_different(data, col1, col2):

    '''

    Check if two rows are identical

    data: dataframe

    col1: str

    col2: str

    '''

    

    if (sum(data[col1] == data[col2])==0):

        return(f'{col1} and {col2} are row wise different')

    else:

        return(f'at least one row has same value for {col1} and {col2}')

    

# Check that type_1 and type_2 are disjoint

print(are_row_wise_different(pokemon, "type_1", 'type_2'))
graph_1 = pokemon.groupby('type_1').count().sort_values(by = 'name')

index_graph_1 = pokemon.groupby('type_1').count().index



graph_2 = pokemon.groupby('type_2').count().sort_values(by = 'name')

index_graph_2 = pokemon.groupby('type_2').count().index
fig = go.Figure(

    data=[go.Bar(x = index_graph_1, 

                 y=graph_1['name'])],

    layout_title_text="First type distribution",

)



fig.show()



fig = go.Figure(

    data=[go.Bar(x = index_graph_2, 

                 y=graph_2['name'])],

    layout_title_text="Second type distribution"

)



fig.show()
fig = make_subplots(rows=1, 

                    cols=2, 

                    specs=[[{'type':'domain'}, 

                            {'type':'domain'}]])



fig.add_trace(go.Pie(labels=index_graph_1, 

                     values=graph_1['name'], 

                     name='Pie chart of first type'),

              1, 1)



fig.add_trace(go.Pie(labels=index_graph_2, 

                     values=graph_2['name'], 

                     name='Pie chart of second type'),

              1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.3, 

                  hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Pie charts of First and Second Type")



fig.show()
# Select data

columns = ['attack', 'hp', 'defense', 'height_m', 'weight_kg', 'sp_attack', 'sp_defense', 'speed']

df = pokemon[columns].copy()



# Normalize data for better readability

normalized_df=(df-df.min())/(df.max()-df.min())
def radar_chart(pokemon_1_index, pokemon_2_index):

    '''

    Print radarchart of two pokemons

    pokemon_1_index: int, index of pokemon in 'normalized_df'

    pokemon_2_index: int, index of pokemon in 'normalized_df'

    '''

    

    fig = go.Figure()



    fig.add_trace(go.Scatterpolar(

          r=normalized_df.loc[pokemon_1_index,:].tolist(),

          theta=columns,

          fill='toself',

          name=pokemon.loc[pokemon_1_index,'name']

    ))

    

    fig.add_trace(go.Scatterpolar(

          r=normalized_df.loc[pokemon_2_index,:].tolist(),

          theta=columns,

          fill='toself',

          name=pokemon.loc[pokemon_2_index,'name']

    ))



    fig.update_layout(

      polar=dict(

        radialaxis=dict(

          visible=True,

          range=[0, 1]

        )),

      showlegend=True

    )

    

    fig.update_layout(

        title="Radar Chart: "+pokemon.loc[pokemon_1_index,'name']+" VS "+pokemon.loc[pokemon_2_index,'name'])

    

    fig.show()
radar_chart(pokemon_1_index = 100, pokemon_2_index = 97)
def cat_total_points(row):

    '''

    Create bins on total_points column

    '''

    

    if row.total_points <300:

        return 'Weakest'

    elif (row.total_points >= 300) & (row.total_points < 600):

        return 'Intermediate'

    else:

        return 'Strong'



# Create bins on total_points column

pokemon['cat_total_points'] = pokemon.apply(cat_total_points, axis='columns')
fig = go.Figure()

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Weakest','height_m'], 

                     name='Weakest',

                marker_color = 'indianred'))

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Intermediate','height_m'], 

                     name='Intermediate',

                marker_color = 'lightseagreen'))

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Strong','height_m'], 

                     name='Strong',

                marker_color = 'mediumpurple'))

fig.update_traces(boxpoints='all', jitter=0)

fig.update_layout(

    title="Height distribution")

fig.show()



fig = go.Figure()

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Weakest','weight_kg'], 

                     name='Weakest',

                marker_color = 'indianred'))

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Intermediate','weight_kg'], 

                     name='Intermediate',

                marker_color = 'lightseagreen'))

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Strong','weight_kg'], 

                     name='Strong',

                marker_color = 'mediumpurple'))

fig.update_traces(boxpoints='all', jitter=0)

fig.update_layout(

    title="Weight distribution")

fig.show()



fig = go.Figure()

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Weakest','speed'], 

                     name='Weakest',

                marker_color = 'indianred'))

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Intermediate','speed'], 

                     name='Intermediate',

                marker_color = 'lightseagreen'))

fig.add_trace(go.Box(y=pokemon.loc[pokemon['cat_total_points']=='Strong','speed'], 

                     name='Strong',

                marker_color = 'mediumpurple'))

fig.update_traces(boxpoints='all', jitter=0)

fig.update_layout(

    title="Speed distribution")



fig.show()
for colm in ['weight_kg', 'height_m', 'speed']:

    find_min_and_max(colm)
columns = ['height_m', 'weight_kg', 'total_points']



fig = px.density_contour(pokemon[columns], 

                         x=np.log(pokemon['height_m']), 

                         y=np.log(pokemon['weight_kg']), 

                         marginal_x="histogram", 

                         marginal_y="histogram")



fig.update_layout(

    title="Two dimension density plot",

    xaxis_title="log(height_m)",

    yaxis_title="log(weight_kg)")



fig.show()
data_to_consider = pokemon[['hp','attack','defense','total_points','cat_total_points']].copy()

data_to_consider = data_to_consider.dropna(0)
fig = px.scatter_ternary(data_to_consider, 

                         a="hp", 

                         b="attack", 

                         c="defense",

                         color="cat_total_points", 

                         size="total_points", 

                         size_max=15)



fig.update_layout(

    title="Ternary plot")



fig.show()
fig = make_subplots(rows=2, cols=2)



fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Normal', 'hp'], 

                     name='Not legendary',

                marker_color = 'indianred'), 

              row=1, 

              col=1)



fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Legendary', 'hp'], 

                     name = 'Legendary',

                marker_color = 'lightseagreen'), 

              row=1, 

              col=1)



fig.update_layout(title="Health Point, Attack, Defense, Total points Boxplots")

fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default



fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Normal', 'attack'],

                marker_color = 'indianred', 

                     showlegend=False), 

              row=1, 

              col=2)

fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Legendary', 'attack'],

                marker_color = 'lightseagreen', 

                     showlegend=False), 

              row=1, 

              col=2)

fig.update_traces(quartilemethod="exclusive")



fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Normal', 'defense'],

                marker_color = 'indianred', 

                     showlegend=False), 

              row=2, 

              col=1)

fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Legendary', 'defense'],

                marker_color = 'lightseagreen', 

                     showlegend=False), 

              row=2, 

              col=1)

fig.update_traces(quartilemethod="exclusive")



fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Normal', 'total_points'],

                marker_color = 'indianred', showlegend=False), 

              row=2, 

              col=2)

fig.add_trace(go.Box(y=pokemon.loc[pokemon['status']=='Legendary', 'total_points'],

                marker_color = 'lightseagreen', 

                     showlegend=False), 

              row=2, 

              col=2)

fig.update_traces(quartilemethod="exclusive")





fig.show()
fig = px.violin(pokemon, 

                y="hp", 

                color="status", 

                box=True, 

                points="all",

          hover_data=pokemon.columns)

fig.update_layout(title="Pokemon status VS hp")

fig.show()



fig = px.violin(pokemon, 

                y="defense", 

                color="status", 

                box=True, 

                points="all",

          hover_data=pokemon.columns)

fig.update_layout(title="Pokemon status VS defense")

fig.show()



fig = px.violin(pokemon, 

                y="attack", 

                color="status", 

                box=True, 

                points="all",

          hover_data=pokemon.columns)



fig.update_layout(title="Pokemon status VS attack")

fig.show()
legendary = pokemon[(pokemon['status']=='Legendary')].groupby('generation').count()['name']

sub_legendary = pokemon[(pokemon['status']=='Sub Legendary')].groupby('generation').count()['name']

mythical = pokemon[(pokemon['status']=='Mythical')].groupby('generation').count()['name']



special_pokemons = pd.concat([sub_legendary, legendary, mythical], axis = 1)

special_pokemons.columns = ['Sublegendaries', 'Legendaries', 'Mythicals']

special_pokemons['Total'] = special_pokemons['Sublegendaries'] + special_pokemons['Legendaries'] + special_pokemons['Mythicals']
gen = special_pokemons.index.tolist()



fig = go.Figure(data=[

    go.Bar(name='Sublegendaries', x=gen, y=special_pokemons['Sublegendaries']),

    go.Bar(name='Legendaries', x=gen, y=special_pokemons['Legendaries']),

    go.Bar(name='Mythicals', x=gen, y=special_pokemons['Mythicals']),

    go.Bar(name='Total', x=gen, y=special_pokemons['Total'])

])



fig.update_layout(barmode='group', title = 'Special pokemon count per generation',

                  xaxis_title="Generation",

                  yaxis_title="Count")



fig.show()
# Gather is_legendary, is_sub_legendary and is_mythical into a single attribute

pokemon['is_special'] = 0

pokemon.loc[pokemon['status'] != 'Normal', 'is_special'] = 1
columns = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed','weight_kg','height_m', 'status']



fig = px.scatter_matrix(pokemon[columns],

    dimensions=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed','weight_kg','height_m'],

    color="status",

    width=1400, 

    height=1400)

fig.update_layout(title='Scatter matrix')

fig.show()
pokemon['total_points_cat'] = 100

pokemon.loc[(pokemon['total_points'] > 100) & (pokemon['total_points'] <= 200), 'total_points_cat'] = 200

pokemon.loc[(pokemon['total_points'] > 200) & (pokemon['total_points'] <= 300), 'total_points_cat'] = 300

pokemon.loc[(pokemon['total_points'] > 300) & (pokemon['total_points'] <= 400), 'total_points_cat'] = 400

pokemon.loc[(pokemon['total_points'] > 400) & (pokemon['total_points'] <= 500), 'total_points_cat'] = 500

pokemon.loc[(pokemon['total_points'] > 500) & (pokemon['total_points'] <= 600), 'total_points_cat'] = 600

pokemon.loc[(pokemon['total_points'] > 600) & (pokemon['total_points'] <= 700), 'total_points_cat'] = 700

pokemon.loc[(pokemon['total_points'] > 700) & (pokemon['total_points'] <= 800), 'total_points_cat'] = 800

pokemon.loc[(pokemon['total_points'] > 800), 'total_points_cat'] = 900



pokemon['hp_cat'] = 50

pokemon.loc[(pokemon['hp'] > 50) & (pokemon['hp'] <= 75), 'hp_cat'] = 75

pokemon.loc[(pokemon['hp'] > 75) & (pokemon['hp'] <= 100), 'hp_cat'] = 100

pokemon.loc[(pokemon['hp'] > 100) & (pokemon['hp'] <= 130), 'hp_cat'] = 130

pokemon.loc[(pokemon['hp'] > 130), 'hp_cat'] = 150



pokemon['attack_cat'] = 50

pokemon.loc[(pokemon['attack'] > 50) & (pokemon['attack'] <= 75), 'attack_cat'] = 75

pokemon.loc[(pokemon['attack'] > 75) & (pokemon['attack'] <= 100), 'attack_cat'] = 100

pokemon.loc[(pokemon['attack'] > 100) & (pokemon['attack'] <= 150), 'attack_cat'] = 150

pokemon.loc[(pokemon['attack'] > 150), 'attack_cat'] = 160



pokemon['defense_cat'] = 50

pokemon.loc[(pokemon['defense'] > 50) & (pokemon['defense'] <= 75), 'defense_cat'] = 75

pokemon.loc[(pokemon['defense'] > 75) & (pokemon['defense'] <= 100), 'defense_cat'] = 100

pokemon.loc[(pokemon['defense'] > 100) & (pokemon['defense'] <= 150), 'defense_cat'] = 150

pokemon.loc[(pokemon['defense'] > 150), 'defense_cat'] = 175
columns = ['status', 'hp_cat','attack_cat','defense_cat','total_points_cat']



data = pokemon[columns].copy()

fig = px.parallel_categories(data, color="total_points_cat", color_continuous_scale=px.colors.sequential.Inferno)

fig.update_layout(title='Funnel for special pokemons through main categorised characteristics')

fig.show()
per_gen_pokemon = pokemon.groupby('generation').mean()[['total_points','hp','attack','defense']]
fig = go.Figure()

fig.add_trace(go.Bar(x=per_gen_pokemon.index, y=per_gen_pokemon['hp'],

                    name='Health Points'))

fig.add_trace(go.Bar(x=per_gen_pokemon.index, y=per_gen_pokemon['defense'],

                    name='Defense'))

fig.add_trace(go.Bar(x=per_gen_pokemon.index, y=per_gen_pokemon['attack'],

                    name='Attack'))

fig.add_trace(go.Bar(x=per_gen_pokemon.index, y=per_gen_pokemon['total_points'],

                    name='Total Points'))

fig.update_layout(barmode='stack', title = 'Stacked barplot of characteristic stats aggregated by generation')

fig.show()
pokemon.head()
# Number of missing values in each column of training data

pokemon.isnull().sum()
pokemon = pokemon.drop(['type_2','ability_2','ability_hidden','catch_rate','base_friendship','base_experience','egg_type_2','percentage_male'], axis = 1)

pokemon = pokemon.dropna()
# All categorical columns

object_cols = [col for col in pokemon.columns if pokemon[col].dtype == "object"]



object_unique = list(map(lambda col: pokemon[col].unique(), object_cols))

d = dict(zip(object_cols, object_unique))
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if len(pokemon[col].unique()) < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
pd.value_counts(pokemon.growth_rate.values)
le = LabelEncoder()

le.fit(['Medium Slow',"Slow",'Fluctuating', "Medium Fast", 'Fast','Erratic'])

pokemon.growth_rate = le.transform(pokemon.growth_rate)



pokemon['is_legendary'] = 0

pokemon['is_sub_legendary'] = 0

pokemon['is_mythical'] = 0

pokemon.loc[pokemon['status']=='Legendary', 'is_legendary'] = 1

pokemon.loc[pokemon['status']=='Sub Legendary', 'is_sub_legendary'] = 1

pokemon.loc[pokemon['status']=='Mythical', 'is_mythical'] = 1
pokemon = pokemon.drop(['pokedex_number','name','status','hp_cat', 'attack_cat','cat_total_points','is_special','total_points_cat','ability_1', 'type_1', 'egg_type_1', 'species'], axis = 1)
pokemon.columns
# Shuffling data

pokemon = pokemon.sample(frac=1)



# Split data and target

X = pokemon.drop('total_points', axis = 1)

y = pokemon['total_points']
# Split into train and validation set

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
model_rf = RandomForestRegressor(oob_score = True,

                                 random_state=0)



model_rf.fit(train_X, train_y)

preds = model_rf.predict(val_X)



print(mean_absolute_error(val_y, preds))
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(model_rf.score(train_X, train_y), 

                                                                                             model_rf.oob_score_,

                                                                                             model_rf.score(val_X, val_y)))
features = train_X.columns

importances = model_rf.feature_importances_

indices = np.argsort(importances)



fig = go.Figure(go.Bar(

            x=importances[indices],

            orientation='h'))



fig.update_layout(

    yaxis = dict(

        tickmode = 'array',

        tickvals = list(range(len(indices))),

        ticktext = [features[i] for i in indices]

    )

)



fig.update_layout(title="Feature importance")

fig.show()