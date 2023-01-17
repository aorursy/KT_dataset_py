#Importing Libraries

import pandas as pd

import numpy as np

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go



# Make plotly work with Jupyter notebook

init_notebook_mode(connected = True)
#Importing dataset

epl = pd.read_csv('../input/epldata_final.csv')
epl.info()
epl['club'].describe()
clubs = tuple(set(epl['club']))

print(clubs)
value = []

for club in clubs:

    value.append(sum(epl['market_value'].loc[epl['club']==club]))
keys= clubs

values=value



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Market Value of players of each club")

})
average_age = []

for club in clubs:

    average_age.append(np.mean(epl['age'].loc[epl['club']==club]))
keys= clubs

values=average_age



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Average Age")

})
country, counts = np.unique(epl['nationality'], return_counts=True)
keys= country

values=counts



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Nationality of Players")

})
c_value = []

for c in country:

    c_value.append(sum(epl['market_value'].loc[epl['nationality']==c]))
keys= country

values=c_value



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Market Value vs Nationality")

})
c_value_mean = []

for c,v in zip(c_value,counts):

    c_value_mean.append(c/v)
keys= country

values=c_value_mean



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Mean Market Value vs Nationality")

})
c_value_median = []

for c in country:

    c_value_median.append(np.median(epl['market_value'].loc[epl['nationality']==c]))
keys= country

values=c_value_median



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Median Market Value vs Nationality")

})
c_value_max = []

for c in country:

    c_value_max.append(max(epl['market_value'].loc[epl['nationality']==c]))
keys= country

values=c_value_max



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Maximum Market Value vs Nationality")

})
ages = tuple(set(epl['age']))

age_value_mean = []

age_value_median = []

age_value_max = []

for age in ages:

    age_value_mean.append(np.mean(epl['market_value'].loc[epl['age']==age]))

    age_value_median.append(np.median(epl['market_value'].loc[epl['age']==age]))

    age_value_max.append(max(epl['market_value'].loc[epl['age']==age]))
keys= ages

values=age_value_mean



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Mean Market Value vs Age")

})
keys= ages

values=age_value_median



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Median Market Value vs Age")

})
keys= ages

values=age_value_max



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Maximum Market Value vs Age")

})
positions, p_counts = np.unique(epl['position'], return_counts=True)

position_value_mean = []

position_value_median = []

position_value_max = []

for p in positions:

    position_value_mean.append(np.mean(epl['market_value'].loc[epl['position']==p]))

    position_value_median.append(np.median(epl['market_value'].loc[epl['position']==p]))

    position_value_max.append(max(epl['market_value'].loc[epl['position']==p]))
keys= positions

values=position_value_mean



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Mean Market Value vs Position")

})
keys= positions

values=position_value_median



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Median Market Value vs Position")

})
keys= positions

values=position_value_max



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Maximum Market Value vs Position")

})
position_age_mean = []

position_age_median = []

position_age_max = []

for p in positions:

    position_age_mean.append(np.mean(epl['age'].loc[epl['position']==p]))

    position_age_median.append(np.median(epl['age'].loc[epl['position']==p]))

    position_age_max.append(max(epl['age'].loc[epl['position']==p]))
keys= positions

values=position_age_mean



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Mean Age vs Position")

})
keys= positions

values=position_age_median



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Median Age vs Position")

})
keys= positions

values=position_age_max



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Maximum Age vs Position")

})
keys= positions

values=p_counts



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Number of Players in Each Position")

})