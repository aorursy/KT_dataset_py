import plotly.graph_objects as go

import pandas as pd

import csv
# colors for all our categories



colors = {

  'fish': '#05668d',

  'natural': '#00a896',

  'raw material': '#3b3b3b',

  'decor': '#f4a261',

  'bug': '#02c39a',

  'clothing': '#ffb4a2',

  'flower': '#e5989b',

  'tool': '#e76f51',

  'sea life': '#028090',

  'fruit': '#ef476f',

  'furniture': '#bfa07a',

  'bells': '#ffd119',

  'fossil': '#001ddb',

  'item': '#a8a8a8',

  'diy recipe': '#f0b87d',

  'accessories': '#fc5190',

  'flooring': '#70828a',

  'nan': '#e5e5e5'

}

df = pd.read_csv('../input/animalcrossing_final.csv') # read our final csv
# User list comprehension to create a list of lists from Dataframe rows

list_of_rows = [list(row) for row in df.values]



# Dictionary with each item, followd by some stats

items = {}
# first, iterate through csv and tally only things that have been sold using hashmap

for row in list_of_rows:

  if row[3] != 'bought':

    color = colors[row[5]] if row[5] in colors else '#e5e5e5'

    if row[1] not in items:

      items[row[1]] = [row[1], row[2], row[6], row[5], color, row[4]]

    else:

      items[row[1]][1] += row[2] # if we can collect something from two methods, this ignores that :\
# convert dictionary to a csv that can be plotted

# probably not the best way to go about this - could have just looked at the dataframe but me dum



csv_columns = ['item','quantity','value', 'category', 'color', 'acquisition']

with open('parseddata.csv', 'w') as f:  # Just use 'w' mode in 3.x

    writer = csv.writer(f)

    writer.writerow(csv_columns)

    for data in items.values():

      writer.writerow(data)
data= pd.read_csv("parseddata.csv") # all data excluding things we bought

data['category'].unique() # to make sure all our colors match
fig = go.Figure()



fig.add_trace(go.Scatter(x=data['quantity'],

                                y=data['value'],

                                mode='markers',

                                marker_color=data['color'],

                                text=data['item'],

                                name='yo')) # hover text goes here



fig.update_layout(title='Volume Collected v.s. Value')

fig.show()
# an easier way to make this plot?

# https://plotly.com/python-api-reference/generated/plotly.express.scatter.html

import plotly.express as px



fig1 = px.scatter(data, x = 'quantity', y = 'value', color = 'category', color_discrete_map = colors, hover_name = 'item', title='Volume Collected v.s. Value')

fig1.show()
# now, let's make a plot to see how many items we collected by category and cumulative value



import numpy as np



categories = []

quantities = []

cumulative_values = []

                       

for category, df_category in data.groupby('category'):

    

    # calculate cumulative value

    all_earnings = [int(quantity)*int(value.replace(',', '')) for quantity, value in zip(df_category['quantity'], df_category['value'])]

    cumulative_value = sum(all_earnings)

    

    categories.append(category) 

    quantities.append(df_category['quantity'].sum())

    cumulative_values.append(cumulative_value)
#plot everything

fig2 = go.Figure()



# iterate through each trace

fig2.add_trace(go.Bar(

    x=categories,

    y=quantities,

    name='Number of Items Collected',

    marker_color='#e5989b'

))

fig2.add_trace(go.Bar(

    x=categories,

    y=cumulative_values,

    name='Cumulative Value',

    marker_color='#ef476f'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig2.update_layout(barmode='group', xaxis_tickangle=-45)

fig2.update_layout(title='Number of Items and their Cumulative Value by Category')

fig2.show() # fail
# never mind, let's just plot them individually

fig3 = go.Figure()



# iterate through each trace

fig3.add_trace(go.Bar(

    x=categories,

    y=quantities,

    name='Number of Items Collected',

    marker_color='#e5989b'

))

fig3.update_layout(barmode='group', xaxis_tickangle=-45)

fig3.update_layout(title='Number of Items Collected by Category')

fig3.show() # fail



fig4 = go.Figure()



# iterate through each trace

fig4.add_trace(go.Bar(

    x=categories,

    y=cumulative_values,

    name='Cumulative Value',

    marker_color='#ef476f'

))

fig4.update_layout(barmode='group', xaxis_tickangle=-45)

fig4.update_layout(title='Cumulative Value of Items Collected by Category')

fig4.show()
# now, let's see how many items and their value we bought, sold, kept, and donated



types_of_things_happened = [] # great variable names 

number_of_things_collected = []

cumulative_earnings = []

                       

for what_happened, df_what_happened in df.groupby('what happened?'):

    types_of_things_happened.append(what_happened)

    number_of_things_collected.append(df_what_happened['quantity'].sum())

    all_earnings = [int(quantity)*int(value.replace(',', '')) for quantity, value in zip(df_what_happened['quantity'], df_what_happened['amount worth each'])]

    multiplier = -1 if what_happened in ('bought','donated') else 1 # not sure if this is a legit thing to do

    cumulative_earnings.append(sum(all_earnings)*multiplier)

    

#plot everything

fig5 = go.Figure()



# iterate through each trace

fig5.add_trace(go.Bar(

    x=types_of_things_happened,

    y=number_of_things_collected,

    name='Number of Items Collected',

    marker_color='#001ddb'

))

fig5.add_trace(go.Bar(

    x=types_of_things_happened,

    y=cumulative_earnings,

    name='Cumulative Value',

    marker_color='#02c39a'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig5.update_layout(barmode='group', xaxis_tickangle=-45)

fig5.update_layout(title='Number of Items and their Cumulative Value by Category')

fig5.show() # fail
fig6 = go.Figure()



# iterate through each trace

fig6.add_trace(go.Bar(

    x=types_of_things_happened,

    y=number_of_things_collected,

    name='Number of Items Collected',

    marker_color='#001ddb'

))

fig6.update_layout(barmode='group', xaxis_tickangle=-45)

fig6.update_layout(title='Number of Items Across Acquisition Categories')

fig6.show() # fail



fig7 = go.Figure()



# iterate through each trace

fig7.add_trace(go.Bar(

    x=types_of_things_happened,

    y=cumulative_earnings,

    name='Cumulative Value',

    marker_color='#02c39a'

))

fig7.update_layout(barmode='group', xaxis_tickangle=-45)

fig7.update_layout(title='Net Gains Across Acquisition Categories')

fig7.show() # fail
# what is the best means of making money?

# let's compare different means of acquiring items and their average payout

# also, categories and their average payout



acquisition_methods = []

categories = []

                       

for what_happened, df_what_happened in data.groupby("acquisition"):

    all_values = []

    for quantity, value in zip(df_what_happened['quantity'], df_what_happened['value']):

        if quantity == 1:

            acquisition_methods.append([what_happened, int(value.replace(',', ''))])

        else:

            acquisition_methods.extend(quantity*[[what_happened, int(value.replace(',', ''))]])

    

# also, categories and their average payout

for category, df_category in data.groupby("category"):

    all_values = []

    for quantity, value in zip(df_category['quantity'], df_category['value']):

        if quantity == 1:

            categories.append([category, int(value.replace(',', ''))])

        else:

            categories.extend(quantity*[[category, int(value.replace(',', ''))]])
#convert to dataframes

acquisition_df = pd.DataFrame(acquisition_methods, columns=['method', 'value'])

category_df = pd.DataFrame(categories, columns=['category', 'value'])
fig8 = px.box(acquisition_df, x='method', y='value')

fig8.update_layout(title='Variance of Item Value Across Acquisition Methods')

fig8.show()
fig9 = px.box(category_df, x='category', y='value')

fig9.update_layout(title='Variance of Item Value Across Categories')

fig9.show()
# that's it! hope that was fun

print('❤️')