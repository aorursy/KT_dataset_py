import numpy as np

import pandas as pd

import math # Mostly for testing if specific values are NaN



import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as plotly # Interactive plotting library

import plotly.graph_objs as go # Plotly's graphing objects

from plotly import tools # Plotly's tools, such as for making figures/subplots



# Initialize plotly in offline mode

plotly.init_notebook_mode(connected=True)



import zipfile # For extraction of datasets
def load_data_into_df(csv_list, data_folder):

    df_dict = {}

    

    for file_name in csv_list:

        # Strip ".csv" from file name to form key

        key = file_name[:-4]

        df_dict[key] = pd.read_csv(data_folder + '/' + file_name)

    

    return df_dict
data_folder = '../input'

csv_list = ['plots.csv', 'suspects.csv']

df_dict = load_data_into_df(csv_list, data_folder)



dict_keys = list(df_dict.keys())
# View data info, columns, and shape

for key in dict_keys:

    print("DATASET: ", key)

    print("INFO: ", df_dict[key].info())

    print("COLUMNS: ", df_dict[key].columns)

    print("SHAPE: ", df_dict[key].shape)

    print("\n")
# View data heads

for key in dict_keys:

    print("DATASET: ", key)

    print(df_dict[key].head(3))

    print("\n")
# Loop through all imported dataframes

for df in df_dict.values():

    

    # Loop through all columns of the current dataframe

    for column in list(df.columns):

        

        # If the column has the word 'date' anywhere in it, convert to Datetime

        if 'date' in column:

            df[column] = pd.to_datetime(df[column])
starting_nas = pd.isnull(df_dict['plots']['plot_ID']).any().any()



for index, row in df_dict['plots'].iterrows():

    plot_df = pd.DataFrame([row['plot_ID']])

    plot_is_null = pd.isnull(plot_df).any().any()

    if (plot_is_null):

        df_dict['plots'] = df_dict['plots'].set_value(

            index=index, col='plot_ID', value=index

        )
# Build a lookup dictionary with the plot name as the key, and its ID as the value

unique_plots = df_dict['plots'][['plot_ID', 'plot_name']].drop_duplicates()

name_to_id = {row[1]['plot_name']: row[1]['plot_ID']

              for row in unique_plots.iterrows()

             }



# Get a working dictionary consisting only of suspects with no Plot ID,

# and return only the plot_ID, terror_plot, and terror_plot2 columns

curr_df = df_dict['suspects'][

    df_dict['suspects']['plot_ID'].isnull()

][['plot_ID', 'terror_plot', 'terror_plot2']]



def id_lookup(name):

    if (name in name_to_id):

        return name_to_id[name]

    else:

        return np.NaN



for index, row in curr_df.iterrows():

    id1 = id_lookup(row['terror_plot'])

    id2 = id_lookup(row['terror_plot2'])

    

    if (math.isnan(id2)):

        new_id_value = id1

    else:

        new_id_value = str(id1) + ',' + str(id2)

    

    # Assign the new ID value

    df_dict['suspects'] = df_dict['suspects'].set_value(

        index=index, col='plot_ID', value=new_id_value

    )
df = df_dict['plots']

df['victims_wounded'] = df['victims_wounded'].fillna(0)

df['victims_killed'] = df['victims_killed'].fillna(0)



def calc_casualties(x):

    x['casualties'] = x['victims_wounded'] + x['victims_killed']

    return x



df_dict['plots'] = df.apply(calc_casualties, axis=1)
df = df_dict['plots']

df_dict['plots']['year'] = df['plot_name'].apply(lambda x: x[0:4])
df = df_dict['plots']

plot_types = set(df['plot_status'].dropna())



def plot_status_evaluate(x):

    if x['plot_status'] not in plot_types:

        if(x['casualties'] > 0):

            value = 'Not Prevented'

        else:

            value = 'Prevented'

        x['plot_status'] = value

    return x



df_dict['plots'] = df.apply(plot_status_evaluate, axis=1)

print(df.pivot_table(index='plot_status', values=['plot_ID'], aggfunc='count'))
df = df_dict['plots']

df = df.sort_values('year')



trace = go.Histogram2d(

    x=df['year'],

    nbinsx=(17), # Bin quarterly instead of the default yearly

    xgap=3,

    

    y=df['plot_ideology'],

    ygap=2,

    

    colorscale='YIOrRd',

    reversescale=True,

    colorbar=dict(

        title='Number of Plots'

    )

)



layout = go.Layout(

    title="All Plots Over Time by Ideology",

    xaxis=go.XAxis(

        title='Time'

    ),

    yaxis=go.YAxis(

        title='Ideology'

    )

)



data = [trace]

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots']

df = df.sort_values('attack_date')



trace = go.Histogram2d(

    x=df['attack_date'],

    nbinsx=(17 * 4), # Bin quarterly instead of the default yearly

    xgap=3,

    

    y=df['plot_ideology'],

    ygap=2,

    

    colorscale='YIOrRd',

    reversescale=True,

    colorbar=dict(

        title='Number of Plots'

    )

)



layout = go.Layout(

    title="Successful Plots Over Time by Ideology",

    xaxis=go.XAxis(

        title='Time'

    ),

    yaxis=go.YAxis(

        title='Ideology'

    )

)



data = [trace]

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots']

plots = df.pivot_table(index='plot_ideology', values='plot_ID', aggfunc='count')



trace = go.Pie(

    labels=plots.index,

    values=plots

)



layout = go.Layout(

    title='All Plots by Ideology'

)



data = [trace]

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots']



casualties_df = df.pivot_table(index=['plot_ideology'], values=['casualties'], aggfunc='sum')



trace = go.Pie(

    labels=casualties_df.index,

    values=casualties_df['casualties']

)



data = [trace]

layout = go.Layout(title='Total Casualties by Ideology')

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots'].sort_values('year')

ideology_types = set(df['plot_ideology'].dropna())

years = set(df['year'].dropna())

years = sorted(years)



traces = []

for ideo_type in ideology_types:

    curr_df = df[df['plot_ideology'] == ideo_type]

    y_values = [

        curr_df[curr_df['year'] == year]['year'].count()

        for year in years

    ]

    traces.append(go.Scatter(

        x = years,

        y = y_values,

        name=ideo_type

    ))

    

data = traces

layout = go.Layout(

    title = 'Plots Over Time by Ideology',

    yaxis = dict(

        title = 'Number of Plots'

    )

)

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots'].sort_values('attack_date')

ideology_types = set(df['plot_ideology'].dropna())



def scale_sizes(value):

    """Normalizes a list of values around a max and min"""

    max_value = 50

    min_value = 10

    max_in_list = 100

    min_in_list = 1

    return (max_value - min_value) * (value - min_in_list) / (max_in_list - min_in_list) + min_value



traces = []

for ideo_type in ideology_types:

    curr_df = df[df['plot_ideology'] == ideo_type]

    traces.append(go.Scatter(

        x = curr_df['attack_date'],

        y = curr_df['casualties'],

        name=ideo_type,

        mode='markers',

        marker=dict(

            size=curr_df['casualties'].apply(scale_sizes),

            opacity=0.4

        )

    ))

    

data = traces

layout = go.Layout(

    title = 'Casualties Over Time by Ideology',

    yaxis=dict(

        title='Number of Casualties'

    )

)

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots']

df = df.sort_values('attack_date')
# Create a list of all suspect plot IDs, without removing duplicates

plot_values = list(df_dict['suspects']['plot_ID'].dropna())



# For any suspects that are associated with multiple plot IDs, split them and

# store their original index in the list to be deleted later

indexes_to_delete = []

for index, plot_id in enumerate(plot_values):

    plot_id_str = str(plot_id)

    if (',' in plot_id_str):

        joint_ids = plot_id_str.split(',')

        indexes_to_delete.append(index)

        for item in joint_ids:

            plot_values.append(float(item))

    else:

        plot_values[index] = float(plot_id)



# Now that the multiple plot IDs have been split and added to the list separately,

# go through the list from bottom to top and delete their original "joint" ID

for index in reversed(indexes_to_delete):

    del(plot_values[index])

            

# Create a list of all suspect plot IDs, with duplicates removed

plot_set = list(set(plot_values))



# Create a dictionary where the keys are the unique plot IDs, and the values

# are how many times that plot ID appears in the plot_values list

plot_suspects_map = {

    'plot_ID': [float(value) for value in plot_set],

    'suspect_count': [plot_values.count(value) for value in plot_set]

}



# Convert the dictionary to a DataFrame in order and merge it with the main df

plot_suspects_df = pd.DataFrame(plot_suspects_map)



if ('suspect_count' in df.columns): # In case this cell is rerun

    df = df.drop('suspect_count', axis=1)



df = pd.merge(df, plot_suspects_df, on='plot_ID', how='left')

df['suspect_count'] = df['suspect_count'].fillna(0)



# At this point, each row of the main df should have a new column called

# 'suspect_count' with the appropriate values filled in
# The marker size will be determined by the suspect_count. Therefore, to prevent

# marker sizes of 0 (which would render the markers invisible), I create a new

# column that represents the size, with a minimum value of 0.5



# This is the function to apply to the "suspect_count" to generate the marker size

default_scale_size = 10

size_column = 'casualties'



def scale_function(x):

    test_series = pd.Series([x])

    if (not test_series.any):

        print('Nan value found')

        return default_scale_size

    

    max_value = 75

    min_value = 20

    scaled_value = (max_value - min_value) * (x - 0) / (30 - 0) + min_value

    return max(min_value, min(max_value, scaled_value))



df['marker_size'] = df[size_column].map(scale_function)
ideology_types = set(df['plot_ideology'])

traces = []



for index, ideo_type in enumerate(ideology_types):

    current_df = df[df['plot_ideology'] == ideo_type]

    traces.append(

        go.Scatter3d(

            x=current_df['plot_status'],

            y=current_df['year'].fillna(min(df['attack_date'])),

            z=current_df['suspect_count'],

            text=current_df['plot_name'],

            name=ideo_type,

            mode='markers',

            marker=dict(

                size=current_df['marker_size'],

                sizemode='diameter',

                sizeref=2.5,

                line=dict(

                    width=0.5

                ),

                opacity=0.9

            )

        )

    )



layout = go.Layout(scene=dict(

    xaxis=dict(title='Prevention Status'),

    yaxis=dict(title='Year'),

    zaxis=dict(title='Suspects Involved')),

    

    margin=dict(l=0, r=0, b=0, t=0),

)



data = traces

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['plots']



year_list = sorted(set(df['year'].dropna().values))

ideo_list = set(df['plot_ideology'].dropna().values)

status_list = set(df['plot_status'].dropna().values)





data_by_year_status = df.pivot_table(

    index=['year', 'plot_ideology', 'plot_status'],

    values=['plot_ID'],

    aggfunc='count',

    fill_value=0

)



data_by_year = df.pivot_table(

    index=['year', 'plot_ideology'],

    values=['plot_ID'],

    aggfunc='count',

    fill_value=0

)



traces = []

for ideo in ideo_list:

    query1_string = 'plot_ideology == "{}"'.format(ideo)

    query2_string = 'plot_ideology == "{}" & plot_status == "Not Prevented"'.format(ideo)

    

    attempt_df = data_by_year.query(query1_string)

    success_df = data_by_year_status.query(query2_string)

    

    attempt_labels = [x for x in attempt_df.index.get_level_values('year')]

    success_labels = [x for x in success_df.index.get_level_values('year')]

    

    attempt_values = attempt_df['plot_ID'].values

    success_values = success_df['plot_ID'].values

    

    attempt_dict = {label: key for label, key in zip(attempt_labels, attempt_values)}

    success_dict = {label: key for label, key in zip(success_labels, success_values)}

    

    y_values = []

    for year in year_list:

        success = success_dict.get(year, 0.0)

        attempt = attempt_dict.get(year, 0.0)

        if (attempt > 0):

            y_values.append(success / attempt)

        else:

            y_values.append(None)



    traces.append(go.Scatter(

        x=year_list,

        y=y_values,

        name=ideo,

        line=dict(

            width=1,

            shape='spline'

        ),

        marker=dict(

            size=8

        )

    ))

        

layout = go.Layout(

    title='Plot Success Rate Over Time',

    yaxis=dict(

        title='% Plots Successful',

        tickformat='%'

    )

)

data = traces

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['suspects']



trace_age_gender=[]

gender_types = list(set(df['sex']))

for gender in gender_types:

    gender_df = df[df['sex'] == gender]

    trace_age_gender.append(

        go.Box(

            y=gender_df['age'],

            name=gender,

            showlegend=False,

            boxpoints='all',

            jitter=0.3,

            whiskerwidth=0.5,

            marker=dict(

                size=3

            )

        )

    )



layout = go.Layout(

    title='Age and Gender',

    xaxis=dict(

        title='Gender'

    ),

    yaxis=dict(

        title='Age'

    )

)



data = trace_age_gender

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['suspects']



gender_types = set(df['sex'].dropna())

marital_types = set(df['marital_status'].dropna())

traces = []



marital_tuple = [

    (

        marital_status, # Key / Marital Status

        df[df['marital_status'] == marital_status]['age'], # Values

        df[df['marital_status'] == marital_status]['age'].count() # Count

    ) for marital_status in marital_types

]



marital_tuple = sorted(marital_tuple, key=lambda x:x[2], reverse=True)



for marital_status, values, _ in marital_tuple:

    traces.append(go.Box(

        name = marital_status,

        y = values,

        boxpoints='all',

        jitter=1,

        pointpos=0,

        orientation='v',

        showlegend=False,

        marker=dict(

            size=3

        ),

        line=dict(

            width=1

        )

        

    ))



layout = go.Layout(

    title='Marital Status by Age and Gender',

    xaxis=dict(title='Marital Status'),

    yaxis=dict(title='Age')

)



data = traces

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
df = df_dict['suspects']
for index, row in df.iterrows():

    plot_id_str = str(row['plot_ID'])

    plot_ids = plot_id_str.split(',')

        

    for plot_id in plot_ids:

        plot_id = float(plot_id.split('.')[0])

        df.set_value(col='plot_ID', index=index, value=plot_id)
ideo_df = df_dict['plots'][['plot_ID', 'plot_ideology']]



if ('plot_ideology' in df.columns): # In case this cell is rerun

    df = df.drop('plot_ideology', axis=1)



df = pd.merge(left=df, right=ideo_df, on='plot_ID', how='left')
ideology_types = list(set(df['plot_ideology'].dropna()))

citizenship_types = list(set(df['citizenship_status'].dropna()))



total_series = df['citizenship_status'].dropna().value_counts()

total_counts = { t: total_series[t] for t in citizenship_types }



ideo_dict = {

    ideo_type: [(

        t,

        df[(df['plot_ideology'] == ideo_type) & (df['citizenship_status'] == t)]['citizenship_status'].count(),

        total_counts[t]

    ) for t in citizenship_types]

    for ideo_type in ideology_types

}



# Shape data for cumulative distribution

total_counts_tuple = ((x, total_counts[x]) for x in total_counts.keys())

total_counts_tuple = sorted(total_counts_tuple, key=lambda x:x[1], reverse=True)

cumulative_labels = []

cumulative_values = []

total_suspects = sum([total_counts[t] for t in citizenship_types])

cumulative_total = 0

for item in total_counts_tuple:

    cumulative_total += item[1]

    cumulative_labels.append(item[0])

    cumulative_values.append(cumulative_total / total_suspects)
traces = []



for ideo_type in ideology_types:

    ideo_tuple = sorted(ideo_dict[ideo_type], key=lambda x:x[2], reverse=True)

    traces.append(go.Bar(

        x = [x[0] for x in ideo_tuple],

        y = [x[1] for x in ideo_tuple],

        name = ideo_type

    ))



traces.append(go.Scatter(

    x=cumulative_labels,

    y=cumulative_values,

    name='Cumulative Distribution',

    yaxis='y2',

    showlegend=False

))    



layout = go.Layout(

    title='Citizenship Types',

    barmode='stack',

    yaxis=dict(

        title='Number of Suspects',

        range=[0, 200]

    ),

    yaxis2=dict(

        title='Cumulative Percentage of Suspects',

        side='right',

        overlaying='y',

        range=[0, 1],

        tick0=0,

        dtick=0.25,

        tickformat='%'

    )

)



data = traces

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)
contact_priority = {'None': 0, 'Ties': 1, 'Contact': 2}

plot_contact_map = {}



for row in zip(df_dict['suspects']['plot_ID'], df_dict['suspects']['awlaki_contact']):

    plot_id_str = str(row[0])

    contact = str(row[1])

    plot_ids = plot_id_str.split(',')

        

    for plot_id in plot_ids:

        plot_id = plot_id.split('.')[0]

        if(plot_id in plot_contact_map):

            current_value = plot_contact_map[plot_id]

            if (contact_priority[current_value] < contact_priority[contact]):

                plot_contact_map[plot_id] = contact

        else:

            plot_contact_map[plot_id] = contact

            

plot_contact_df = pd.DataFrame(list(plot_contact_map.items()))

plot_contact_df.columns = ['plot_ID', 'awlaki_contact']

plot_contact_df['plot_ID'] = pd.to_numeric(plot_contact_df['plot_ID'], errors='coerce')

plot_contact_df = plot_contact_df.sort_values('plot_ID', ascending=True)
df = df_dict['plots']

df = df[df['plot_ideology'] == 'Jihadist']



df = pd.merge(df, plot_contact_df, on='plot_ID', how='left')
contact_types = list(contact_priority.keys())

prevention_types = list(set(df['plot_status'].dropna()))

y_vals_abs = {} # Absolute numbers

y_vals_totals = np.zeros(len(contact_types)) # Total numbers to divide by to get the percentage

traces = []



for prevention_status in prevention_types:

    y_vals = []

    prevention_df = df[df['plot_status'] == prevention_status]

    for contact_status in contact_types:

        y_vals.append(

            prevention_df[

                prevention_df['awlaki_contact'] == contact_status

            ]['awlaki_contact'].count()

        )

    y_vals_abs[prevention_status] = np.array([float(x) for x in y_vals])

    y_vals_totals = np.add(y_vals_abs[prevention_status], y_vals_totals)



for prevention_status in prevention_types:

    y_vals_rel = np.divide(y_vals_abs[prevention_status], y_vals_totals)

    traces.append(go.Bar(

        x=contact_types,

        y=y_vals_rel,

        textposition='auto',

        text=['{0:.0f}%'.format(y * 100) for y in y_vals_rel],

        textfont=dict(color='white'),

        name=prevention_status

    ))

    

layout = go.Layout(

    title='Plots Prevented and al-Awlaki Contact',

    barmode='stack',

    yaxis=dict(

        title='% of Plots',

        tickformat='%'

    )

)



data = traces

figure = go.Figure(data=data, layout=layout)

plotly.iplot(figure)