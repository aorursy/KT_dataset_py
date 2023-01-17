import pandas as pd
from plotly import plotly
df = pd.read_csv('../input/general_payments.csv', low_memory=False,
                 usecols=['Recipient_State', 'Total_Amount_of_Payment_USDollars'])
df.columns = ['code', 'amount']
# 美国各州缩写及全称字典
#  abbr. and full name
code = {'AL': 'Alabama',
        'AK': 'Alaska',
        'AZ': 'Arizona',
        'AR': 'Arkansas',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'IA': 'Iowa',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'ME': 'Maine',
        'MD': 'Maryland',
        'MA': 'Massachusetts',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MS': 'Mississippi',
        'MO': 'Missouri',
        'MT': 'Montana',
        'NE': 'Nebraska',
        'NV': 'Nevada',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NY': 'New York',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VT': 'Vermont',
        'VA': 'Virginia',
        'WA': 'Washington',
        'WV': 'West Virginia',
        'WI': 'Wisconsin',
        'WY': 'Wyoming'}
code_df = pd.DataFrame.from_dict(code, orient='index').reset_index()
code_df.columns = ['code', 'state']
code_df.head()
desc = lambda x: {'mean': x.mean(), 'sum': x.sum(), 'count': x.count(), 'max': x.max()}
df = df['amount'].groupby(df['code']).apply(desc).unstack().reset_index()
df.head()
df = pd.merge(df, code_df)
df.head()
locations = df['code']
colorscale = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'],
              [0.4, 'rgb(188,189,220)'], [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]
df.sort_values('mean', ascending=False)[['code', 'mean', 'state']].head(10)
p = df['mean']
labels = df['state']

data = [{'type': 'choropleth',
         'colorscale': colorscale,
         'autocolorscale': False,
         'locations': locations,
         'z': p,
         'locationmode': 'USA-states',
         'text': labels,
         'marker': {'line': {'color': 'rgb(255,255,255)', 'width': 2}},
         'colorbar': {'title': 'USD'}}]
layout = {'title': '2013 US Average Amount of Payment Across States', # 2013年美国各州的平均支付金额
          'geo': {'scope': 'usa',
                   'projection': {'type': 'albers usa'},
                   'showlakes': True,
                   'lakecolor': 'rgb(255, 255, 255)'}}
fig = dict(data=data, layout=layout)

# This code can't run on Kaggle, You have to run on your own computer.
# url = plotly.plot(fig, filename='2013 US Average Amount of Payment Across States')
df.sort_values('sum', ascending=False)[['code', 'sum', 'state']].head(10)
p = df['sum']
labels = df['state']

data = [{'type': 'choropleth',
         'colorscale': colorscale,
         'autocolorscale': False,
         'locations': locations,
         'z': p,
         'locationmode': 'USA-states',
         'text': labels,
         'marker': {'line': {'color': 'rgb(255,255,255)', 'width': 2}},
         'colorbar': {'title': 'USD'}}]
layout = {'title': '2013 US Total Amount of Payment Across States',  # 2013年美国各州的支付总额
          'geo': {'scope': 'usa',
                   'projection': {'type': 'albers usa'},
                   'showlakes': True,
                   'lakecolor': 'rgb(255, 255, 255)'}}
fig = dict(data=data, layout=layout)

# This code can't run on Kaggle, You have to run on your own computer.
# url = plotly.plot(fig, filename='2013 US Total Amount of Payment Across States')
df.sort_values('count', ascending=False)[['code', 'count', 'state']].head(10)
p = df['count']
labels = df['state']

data = [{'type': 'choropleth',
         'colorscale': colorscale,
         'autocolorscale': False,
         'locations': locations,
         'z': p,
         'locationmode': 'USA-states',
         'text': labels,
         'marker': {'line': {'color': 'rgb(255,255,255)', 'width': 2}},
         'colorbar': {'title': 'USD'}}]
layout = {'title': '2013 US Number of Payments Across States',  # 2013年美国各州的付款次数
          'geo': {'scope': 'usa',
                   'projection': {'type': 'albers usa'},
                   'showlakes': True,
                   'lakecolor': 'rgb(255, 255, 255)'}}
fig = dict(data=data, layout=layout)

# This code can't run on Kaggle, You have to run on your own computer.
# url = plotly.plot(fig, filename='2013 US Number of Payments Across States')
