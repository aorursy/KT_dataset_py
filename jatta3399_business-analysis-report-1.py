import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

!pip install pywaffle --quiet

from pywaffle import Waffle
df= pd.read_csv("../input/thesparkfoundation/SampleSuperstore.csv")
df
df.shape
df.isnull().any()
quant=df[["Quantity", "Profit"]].groupby(['Quantity'], as_index=False).mean().sort_values(by='Profit', ascending=False)

plt.figure(figsize=(10,8))



sns.barplot(x='Quantity', y='Profit', data=quant)
cat=df[["Category", "Profit"]].groupby(['Category'], as_index=False).mean().sort_values(by='Profit', ascending=False)

plt.figure(figsize=(10,8))



sns.barplot(x='Category', y='Profit', data=cat)
sub_cat=df[["Sub-Category", "Profit"]].groupby(['Sub-Category'], as_index=False).mean().sort_values(by='Profit', ascending=False)

plt.figure(figsize=(20,15))



sns.barplot(x='Sub-Category', y='Profit', data=sub_cat)
numerical = ['Sales','Quantity','Discount','Profit']
df[numerical].hist(bins=25, figsize=(20,10), layout=(2, 2))


categorical = ['Ship Mode','Segment','State','Region','Category','Sub-Category']

fig, ax = plt.subplots(3, 2, figsize=(30, 15))



plt.subplots_adjust(hspace=0.7)

for variable, subplot in zip(categorical, ax.flatten()):

    sns.countplot(df[variable], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
state_code = {'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virginia': 'VA','District of Columbia': 'WA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}

df['state_code'] = df.State.apply(lambda x: state_code[x])
state = df[['Sales', 'Profit', 'state_code']].groupby(['state_code']).sum()





fig = go.Figure(data=go.Choropleth(

    locations=state.index, 

    z = state.Sales, 

    locationmode = 'USA-states', 

    colorscale = 'Reds',

    colorbar_title = 'Sales in USD',

))



fig.update_layout(

    title_text = 'Total State-Wise Sales',

    geo_scope='usa',

    height=800,

)



fig.show()

5
matrix = np.triu(df.corr())

sns.heatmap(df.corr(), annot=True, mask=matrix)
fig, ax = plt.subplots(6, 1, figsize=(20,50))

plt.subplots_adjust(hspace=0.4)

for var, subplot in zip(categorical, ax.flatten()):

    sns.boxplot(x=var, y= 'Profit', data=df, ax=subplot,showfliers=False)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
state['profit_to_sales'] = state['Profit'] / state['Sales']



# adding state name

state_name = {v: k for k, v in state_code.items()}

state['States'] = state.index

state['States'] = state.States.apply(lambda x: state_name[x])



# sorting the dataframe

state = state.sort_values(by = ['profit_to_sales'], ascending=True)
fig = px.bar(state, x = 'profit_to_sales', y = 'States', title = 'PRICE TO SALES RATIO',

            color = 'Profit', color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(

    autosize=False,

    height=1000,

    xaxis = dict(

        tickmode = 'array',

        ticktext = state.States,

        title='Profit to Sales Ratio',

    ),

    yaxis=dict(title='State'),

)

fig.show()
ship_segment = df.groupby(['Segment'])

segment_list = df.Segment.value_counts().index

cat_list = df.Category.value_counts().index



for segment in segment_list:

    seg_shipping = ship_segment.get_group(segment)

    standard, second, first, same = [], [], [], []

    for cat in cat_list:

        count = seg_shipping.groupby(['Category']).get_group(cat)['Ship Mode'].value_counts()

        standard.append(count[0]), second.append(count[1]), first.append(count[2]), same.append(count[3])

        

    fig = go.Figure()

    fig.add_trace(go.Bar(x = cat_list,y = standard,name='Standard Class',marker_color='rgb(137,51,51)'

                        ))

    fig.add_trace(go.Bar(x = cat_list,y = second,name='Second Class',marker_color='rgb(234,84,84)'

                        ))

    fig.add_trace(go.Bar(x = cat_list,y = first,name='First Class',marker_color='rgb(250,127,78)'

                        ))

    fig.add_trace(go.Bar(x = cat_list,y = same,name='Same Day',marker_color='lightsalmon'

                        ))



    fig.update_layout(

        barmode ='group',

        width = 800,

        title = segment.upper(),

        yaxis = dict(title = 'Number of Deliveries'))

    fig.show()
df["Cost"] = df['Sales']/df['Quantity']



#finding profit per sold items



df['Profit'] = df['Profit']/df['Quantity']



#Grouping Data

data_group_one = df[['Ship Mode','Segment','Category','Sub-Category','Cost','Discount','Profit']]

data_group_one = data_group_one.groupby(['Ship Mode','Segment','Category','Sub-Category'],as_index=False).mean()



#Data for first Class & consumer

data_group_1 = data_group_one[data_group_one['Ship Mode'] == 'First Class']

data_group_1 = data_group_1[data_group_1['Segment'] == 'Consumer']



#Data for Same Day & consumer

data_group_2 = data_group_one[data_group_one['Ship Mode'] == 'Same Day']

data_group_2 = data_group_2[data_group_2['Segment'] == 'Consumer']



#Data for Second Class & consumer

data_group_3 = data_group_one[data_group_one['Ship Mode'] == 'Second Class' ]

data_group_3 = data_group_3[data_group_3['Segment'] == 'Consumer']



#Data for Standard Class & consumer

data_group_4 = data_group_one[data_group_one['Ship Mode'] == 'Standard Class']

data_group_4 = data_group_4[data_group_4['Segment'] == 'Consumer']

data_group_1
data_group_2
data_group_3
data_group_4