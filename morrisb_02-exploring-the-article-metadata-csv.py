# To do linear algebra

import numpy as np



# To store data

import pandas as pd



# To create interactive plots

import plotly.graph_objects as go
# Read the csv file

df = pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')



print('{} entries are in the file.'.format(df.shape[0]))

print('{} columns are in the file.'.format(df.shape[1]))

print('\nThese are some sampled entries:')

df.sample(3)
def countUniqueAndNAN(df, column):

    '''

    Counts and prints the number of empty, filled and unique values for a column in a dataframe

    

    Input:

    df - dataframe to use

    column - column to inspect

    

    Output:

    '''

    

    

    # If there are empty values in the column

    if df[column].isna().sum():

    

        # Count the filled and empty values in the column

        tmp_dict = df[column].isna().value_counts().to_dict()



    else:

        tmp_dict = {False: len(df[column]), True: 0}

        

    print('The column "{}" has:\n\n{} filled and\n{} empty values.'.format(column, tmp_dict[False], tmp_dict[True]))





    # Count the unique values in the column

    nunique = df[column].nunique()



    print('\n{} unique values are in the column.'.format(nunique))









def interactiveBarPlot(x, y, column, title):

    '''

    Creates interactive bar plot with x and y data

    

    Input:

    x - data on x axis

    y - data on y axis

    column - column to inspect

    title- title of the plot

    n - number of most counted items to plot

    

    Output:

    '''



    bar = go.Bar(x=x, 

                 y=y, 

                 orientation='h')



    layout = go.Layout(title=title, 

                       xaxis_title='Value Count', 

                       yaxis_title='{}'.format(column))



    fig = go.Figure([bar], layout)

    fig.show()
# Define the column to inspect

column = 'sha'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'source_x'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'title'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Shorten the y-axis labels

y = [i[:30] for i in y]



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'doi'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'pmcid'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'pubmed_id'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Shorten the y-axis labels

y = ['ID: '+str(i)[:30] for i in y]



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'license'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'abstract'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Shorten the y-axis labels

y = [i[:30] for i in y]



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'publish_time'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Convert column to datetime

df_tmp = pd.to_datetime(df['publish_time'].dropna(), errors='coerce').to_frame()



# Set and sort index

df_tmp.set_index('publish_time', inplace=True)

df_tmp.sort_index(inplace=True)



# Resample the data on month basis

df_tmp = df_tmp.resample('M').size().to_frame()



# Filter empty months out

df_tmp = df_tmp[df_tmp[0]!=0]



# Create plot

scatter = go.Scatter(x=df_tmp.index, 

                     y=df_tmp[0], 

                     mode='markers')



layout = go.Layout(title='Number of publications over time', 

                   xaxis_title='Month of publication', 

                   yaxis_title='Number of publications')



fig = go.Figure([scatter], layout)

fig.show()
# Define the column to inspect

column = 'authors'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Shorten the y-axis labels

y = [i[:30] for i in y]



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'journal'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'Microsoft Academic Paper ID'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Shorten the y-axis labels

y = ['ID: '+ str(i)[:30] for i in y]



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'WHO #Covidence'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Shorten the y-axis labels

y = ['WHO: '+ str(i)[:30] for i in y]



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Define the column to inspect

column = 'has_full_text'





# Count empty, filled and unique values for the column

countUniqueAndNAN(df, column)







# Create mapper for empty and filled values

mapper = {False:'Filled Values', True:'Empty Values'}



# Count empty values

tmp_dict = df[column].isna().value_counts().to_dict()



# Split data to x- and y-axis

y, x = zip(*tmp_dict.items())



# Title of the plot

title = 'Counted empty values for the column "{}"'.format(column)



# Plot count of empty values

interactiveBarPlot(x, [mapper[i] for i in y], column, title)







# Number of most common itemsto plot

n = 20



# Get value counts

tmp_data = df[column].value_counts().head(n)



# Get the data

x = tmp_data.values

y = tmp_data.index



# Title of the plot

title = 'Counted category values for the column "{}"'.format(column) + ('' if len(x)<n else ' (Top {})'.format(n))



# Create the plot

interactiveBarPlot(x, y, column, title)
# Copy the dataframe

df_combine = df[['source_x', 'license', 'publish_time', 'journal', 'has_full_text']].copy()



# Convert column to datetime

df_combine['publish_time'] = pd.to_datetime(df_combine['publish_time'], errors='coerce')



# Set and sort index

df_combine.set_index('publish_time', inplace=True)

df_combine.sort_index(inplace=True)



# Clean license column

df_combine['license'] = df_combine['license'].replace(np.nan, '', regex=True).apply(lambda x: x.replace('-', ' ').upper()).replace('', np.nan, regex=True)
scatter = []



column = 'source_x'



# Iterate over all unique sources

for value in df_combine[column].unique():

    

    # Resample the data on month basis

    df_tmp = df_combine[(df_combine[column]==value) & (df_combine.index.notnull())].resample('M').size().to_frame().loc['2000':]

    

    # Filter empty months out

    df_tmp = df_tmp[df_tmp[0]!=0]

    

    # Create scatter

    scatter.append(go.Scatter(x=df_tmp.index, 

                              y=df_tmp[0],

                              name=value))





layout = go.Layout(title='Number of publications over time for the column "{}"'.format(column), 

                   xaxis_title='Month of publication', 

                   yaxis_title='Number of publications')



fig = go.Figure(scatter, layout)

fig.show()
scatter = []



column = 'license'



# Iterate over all unique sources

for value in df_combine[column].unique():

    

    # Resample the data on month basis

    df_tmp = df_combine[(df_combine[column]==value) & (df_combine.index.notnull())].resample('M').size().to_frame().loc['2000':]

    

    # Filter empty months out

    df_tmp = df_tmp[df_tmp[0]!=0]

    

    # Create scatter

    scatter.append(go.Scatter(x=df_tmp.index, 

                              y=df_tmp[0], 

                              name=value))





layout = go.Layout(title='Number of publications over time for the column "{}"'.format(column), 

                   xaxis_title='Month of publication', 

                   yaxis_title='Number of publications')



fig = go.Figure(scatter, layout)

fig.show()
scatter = []



column = 'has_full_text'



# Iterate over all unique sources

for value in df_combine[column].unique():

    

    # Resample the data on month basis

    df_tmp = df_combine[(df_combine[column]==value) & (df_combine.index.notnull())].resample('M').size().to_frame().loc['2000':]

    

    # Filter empty months out

    df_tmp = df_tmp[df_tmp[0]!=0]

    

    # Create scatter

    scatter.append(go.Scatter(x=df_tmp.index, 

                              y=df_tmp[0], 

                              name=value))





layout = go.Layout(title='Number of publications over time for the column "{}"'.format(column), 

                   xaxis_title='Month of publication', 

                   yaxis_title='Number of publications')



fig = go.Figure(scatter, layout)

fig.show()
# Create a pivot table to find the most used license by source_x

df_tmp = df_combine.pivot_table(values='has_full_text', index='license', columns='source_x', aggfunc='size', fill_value=0)



# Compute percentages

df_tmp = df_tmp / df_tmp.sum(axis=0) * 100



df_tmp
# Create a pivot table to find the most used license by journal

df_tmp = df_combine.pivot_table(values='license', index='has_full_text', columns='source_x', aggfunc='size', fill_value=0)



# Compute percentages

df_tmp = df_tmp / df_tmp.sum(axis=0) * 100



df_tmp