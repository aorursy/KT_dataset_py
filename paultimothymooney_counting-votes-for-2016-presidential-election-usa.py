# Import Python Modules

import numpy as np

import pandas as pd

import plotly.express as px

# Read election data

results = pd.read_csv('/kaggle/input/open-elections-data-usa/openelections-data-us-master/openelections-data-us-master/2016/20161108__us__general__president__county.csv')

# Sum election data for each candidate

results = results.groupby("candidate").sum().sort_values('votes',ascending=False).drop('district',axis=1)

# Combine duplicate results

results.loc['Donald J. Trump'] += results.loc['TRUMP, DONALD J '] 

results.loc['Hillary Clinton'] += results.loc['Hillary Rodham Clinton'] 

results.loc['Hillary Clinton'] += results.loc['Hillary Clinton / Tim Kaine']

results.loc['Hillary Clinton'] += results.loc['Hillary Clinton and Tim Kaine']

results.loc['Hillary Clinton'] += results.loc['Hillary Clinton - Tim Kaine']

results.loc['Hillary Clinton'] += results.loc['HILLARY CLINTON'] 

results.loc['Hillary Clinton'] += results.loc['HILLARY RODHAM CLINTON']

results.loc['Hillary Clinton'] += results.loc['HILLARY CLINTON | TIM KAINE']

results.loc['Gary Johnson'] += results.loc['GARY JOHNSON']

results.loc['Gary Johnson'] += results.loc['Gary Johnson / Bill Weld']

results.loc['Gary Johnson'] += results.loc['JOHNSON, GARY E ']

results.loc['Gary Johnson'] += results.loc['Gary Johnson and William Weld']

results.loc['Gary Johnson'] += results.loc['GARY JOHNSON | BILL WELD']

results.loc['Gary Johnson'] += results.loc['Gary Johnson - Bill Weld']

results.loc['Gary Johnson'] += results.loc['Gary  Johnson']

results.loc['Jill Stein'] += results.loc['Jill Stein / Ajamu Baraka']

results.loc['Jill Stein'] += results.loc['JILL STEIN']

results.loc['Jill Stein'] += results.loc['Jill  Stein']

results.loc['Jill Stein'] += results.loc['STEIN, JILL ']

results.loc['Jill Stein and Howie Hawkins'] += results.loc['Jill Stein and Howie Hawkins']

results.loc['Jill Stein and Howie Hawkins'] += results.loc['Jill Stein - Ajamu Baraka']

results = results.drop(['Total','Totals','Hillary Rodham Clinton',

                        'HILLARY CLINTON','Hillary Clinton / Tim Kaine',

                        'HILLARY RODHAM CLINTON','TRUMP, DONALD J ',

                        'HILLARY CLINTON | TIM KAINE','Hillary Clinton - Tim Kaine',

                       'Hillary Clinton and Tim Kaine','GARY JOHNSON',

                       'Gary Johnson / Bill Weld', 'JOHNSON, GARY E ',

                        'Gary Johnson - Bill Weld','Gary  Johnson',

                        'Gary Johnson and William Weld','GARY JOHNSON | BILL WELD',

                       'Jill Stein / Ajamu Baraka','JILL STEIN','Jill  Stein',

                       'STEIN, JILL ','Jill Stein and Howie Hawkins',

                       'Jill Stein - Ajamu Baraka'])

# Combine all candidates that were not in the top 4

results['Candidate'] = results.index

other_candidates = results[4:].sum().values[0]

results.loc['Other'] = [other_candidates, 'Other'] 

list_of_candidates = ['Donald J. Trump','Hillary Clinton','Gary Johnson','Jill Stein','Other']

results = results.loc[results['Candidate'].isin(list_of_candidates)]

results = results.sort_values('votes',ascending='True')

# Preview results

results.head()
# Percentages are good but raw numbers are too large -- revisit this another time



# fig = px.bar(results, x=results.Candidate, y=results.votes,title='Count of Votes in the 2016 Presidential Election (USA)')

# fig.show()
percentages = results

percentages['votes'] = percentages['votes']/percentages['votes'].sum()*100

fig = px.bar(percentages, x=percentages.Candidate, y=percentages.votes,title='Percentage of Votes in the 2016 Presidential Election (USA)')

fig.show()