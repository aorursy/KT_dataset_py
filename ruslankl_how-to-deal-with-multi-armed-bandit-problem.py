import numpy as np

import pandas as pd

from scipy.stats import beta, bernoulli

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from plotly import tools

import random

import math



RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)

init_notebook_mode(connected=True)
def algorithm_performance():

    """

    Function that will show the performance of each algorithm we will be using in this tutorial.

    """

    

    ## calculate how many time each Ad has been choosen

    count_series = pd.Series(index_list).value_counts(normalize=True)

    print('Ad #0 has been shown', count_series[0]*100, '% of the time.')

    print('Ad #1 has been shown', count_series[1]*100, '% of the time.')

    

    print('Total Reward (Number of Clicks):', total_reward) ## print total Reward

    

    x = np.arange (0, n, 1)



    ## plot the calculated CTR for Ad #0

    data1 = go.Scatter(x=x,

                       y=ctr[0],

                       name='Calculated CTR #0',

                       line=dict(color=('rgba(10, 108, 94, .7)'),

                                 width=2))



    ## plot the line with actual CTR for Ad #0

    data2 = go.Scatter(x=[0, n],

                       y=[ACTUAL_CTR[0]] * 2,

                       name='Actual CTR #0 value',

                       line = dict(color = ('rgb(205, 12, 24)'),

                                   width = 1,

                                   dash = 'dash'))



    ## plot the calculated CTR for Ad #1

    data3 = go.Scatter(x=x,

                       y=ctr[1],

                       name='Calculated CTR #1',

                       line=dict(color=('rgba(187, 121, 24, .7)'),

                                 width=2))



    ## plot the line with actual CTR for Ad #0

    data4 = go.Scatter(x=[0, n],

                       y=[ACTUAL_CTR[1]] * 2,

                       name='Actual CTR #1 value',

                       line = dict(color = ('rgb(205, 12, 24)'),

                                   width = 1,

                                   dash = 'dash'))



    ## plot the Regret values as a function of trial number

    data5 = go.Scatter(x=x,

                       y=regret_list,

                       name='Regret')



    layout = go.Layout(title='Simulated CTR Values and Algorithm Regret',

                       xaxis={'title': 'Trial Number'},

                       yaxis1={'title': 'CTR value'},

                       yaxis2={'title': 'Regret Value'}

                       )

    fig = tools.make_subplots(rows=2, cols=1, print_grid=False, shared_yaxes=True, shared_xaxes=True)



    fig.append_trace(data1, 1, 1)

    fig.append_trace(data2, 1, 1)

    fig.append_trace(data3, 1, 1)

    fig.append_trace(data4, 1, 1)

    fig.append_trace(data5, 2, 1)



    fig['layout'].update(layout)

    iplot(fig, show_link=False)
ACTUAL_CTR = [.45, .65]

print('Actual CTR for Ad #0 is:', ACTUAL_CTR[0])

print('Actual CTR for Ad #1 is:', ACTUAL_CTR[1])
## For each alrorithm we will perform 1000 trials

n = 1000
regret = 0 

total_reward = 0

regret_list = [] ## list for collecting the regret values for each impression (trial)

ctr = {0: [], 1: []} ## lists for collecting the calculated CTR 

index_list = [] ## list for collecting the number of randomly choosen Ad



## set the initial values for impressions and clicks 

impressions = [0,0] 

clicks = [0,0]



for i in range(n):    

    

    random_index = np.random.randint(0,2,1)[0] ## randomly choose the value between [0,1]

    index_list.append(random_index) ## add the value to list

    

    impressions[random_index] += 1 ## add 1 impression value for the choosen Ad

    did_click = bernoulli.rvs(ACTUAL_CTR[random_index]) ## simulate if the person clicked on the ad usind Actual CTR value

    

    if did_click:

        clicks[random_index] += did_click ## if person clicked add 1 click value for the choosen Ad

    

    ## calculate the CTR values and add them to list

    if impressions[0] == 0:

        ctr_0 = 0

    else:

        ctr_0 = clicks[0]/impressions[0]

        

    if impressions[1] == 0:

        ctr_1 = 0

    else:

        ctr_1 = clicks[1]/impressions[1]

        

    ctr[0].append(ctr_0)

    ctr[1].append(ctr_1)

    

    ## calculate the regret and reward

    regret += max(ACTUAL_CTR) - ACTUAL_CTR[random_index]

    regret_list.append(regret)

    total_reward += did_click
algorithm_performance()
## save the reward and regret values for future comparison

random_dict = {'reward':total_reward, 

               'regret_list':regret_list, 

               'ads_count':pd.Series(index_list).value_counts(normalize=True)}
e = .05 ## set the Epsilon value

n_init = 100 ## number of impressions to choose the winning Ad

impressions = [0,0]

clicks = [0,0]



for i in range(n_init):

    random_index = np.random.randint(0,2,1)[0]

    

    impressions[random_index] += 1

    did_click = bernoulli.rvs(ACTUAL_CTR[random_index])

    if did_click:

        clicks[random_index] += did_click

        

ctr_0 = clicks[0] / impressions[0]

ctr_1 = clicks[1] / impressions[1]

win_index = np.argmax([ctr_0, ctr_1]) ## select the Ad number with the highest CTR



print('After', n_init, 'initial trials Ad #', win_index, 'got the highest CTR =', round(np.max([ctr_0, ctr_1]), 2), 

      '(Real CTR value is', ACTUAL_CTR[win_index], ').'

      '\nIt will be shown', (1-e)*100, '% of the time.')
regret = 0 

total_reward = 0

regret_list = [] 

ctr = {0: [], 1: []}

index_list = [] 

impressions = [0,0] 

clicks = [0,0]



for i in range(n):    

    

    epsilon_index = random.choices([win_index, 1-win_index], [1-e, e])[0]

    index_list.append(epsilon_index)

    

    impressions[epsilon_index] += 1

    did_click = bernoulli.rvs(ACTUAL_CTR[epsilon_index])

    if did_click:

        clicks[epsilon_index] += did_click

    

    if impressions[0] == 0:

        ctr_0 = 0

    else:

        ctr_0 = clicks[0]/impressions[0]

        

    if impressions[1] == 0:

        ctr_1 = 0

    else:

        ctr_1 = clicks[1]/impressions[1]

        

    ctr[0].append(ctr_0)

    ctr[1].append(ctr_1)

    

    regret += max(ACTUAL_CTR) - ACTUAL_CTR[epsilon_index]

    regret_list.append(regret)

    total_reward += did_click
algorithm_performance()
epsilon_dict = {'reward':total_reward, 

                'regret_list':regret_list, 

                'ads_count':pd.Series(index_list).value_counts(normalize=True)}
regret = 0 

total_reward = 0

regret_list = [] 

ctr = {0: [], 1: []}

index_list = [] 

impressions = [0,0] 

clicks = [0,0]

priors = (1, 1)

win_index = np.random.randint(0,2,1)[0] ## randomly choose the first shown Ad



for i in range(n):    

    

    impressions[win_index] += 1

    did_click = bernoulli.rvs(ACTUAL_CTR[win_index])

    if did_click:

        clicks[win_index] += did_click

    

    ctr_0 = random.betavariate(priors[0]+clicks[0], priors[1] + impressions[0] - clicks[0])

    ctr_1 = random.betavariate(priors[0]+clicks[1], priors[1] + impressions[1] - clicks[1])

    win_index = np.argmax([ctr_0, ctr_1])

    index_list.append(win_index)

    

    ctr[0].append(ctr_0)

    ctr[1].append(ctr_1)

    

    regret += max(ACTUAL_CTR) - ACTUAL_CTR[win_index]

    regret_list.append(regret)    

    total_reward += did_click
## plot the Beta distributions

x = np.arange (0, 1, 0.01)

y = beta.pdf(x, priors[0]+clicks[0], priors[1] + impressions[0] - clicks[0])

y /= y.max() ## normalize



data1 = go.Scatter(x=x,

                   y=y,

                   name='Beta Distribution (Ad #0)',

                   marker = dict(color=('rgba(10, 108, 94, 1)')),

                   fill='tozeroy',

                   fillcolor = 'rgba(10, 108, 94, .7)')



data2 = go.Scatter(x = [ACTUAL_CTR[0]] * 2,

                   y = [0, 1],

                   name = 'Actual CTR #0 Value',

                   mode='lines',

                   line = dict(

                       color = ('rgb(205, 12, 24)'),

                       width = 2,

                       dash = 'dash'))



y = beta.pdf(x, priors[0]+clicks[1], priors[1] + impressions[1] - clicks[1])

y /= y.max()



data3 = go.Scatter(x=x,

                   y=y,

                   name='Beta Distribution (Ad #1)',

                   marker = dict(color=('rgba(187, 121, 24, 1)')),

                   fill='tozeroy',

                   fillcolor = 'rgba(187, 121, 24, .7)')



data4 = go.Scatter(x = [ACTUAL_CTR[1]] * 2,

                   y = [0, 1],

                   name = 'Actual CTR #1 Value',

                   mode='lines',

                   line = dict(

                       color = ('rgb(205, 12, 24)'),

                       width = 2,

                       dash = 'dash'))



layout = go.Layout(title='Beta Distributions for both Ads',

                   xaxis={'title': 'Possible CTR values'},

                   yaxis={'title': 'Probability Density'})



fig = go.Figure(data=[data1, data2, data3, data4], layout=layout)



# fig = tools.make_subplots(rows=1, cols=2, print_grid=False, shared_xaxes=False,

#                           subplot_titles=('Beta Distribution (Ad #0)','Beta Distribution (Ad #1)'))



# fig.append_trace(data1, 1, 1)

# fig.append_trace(data2, 1, 1)

# fig.append_trace(data3, 1, 2)

# fig.append_trace(data4, 1, 2)



# fig['layout'].update(showlegend=False)



iplot(fig, show_link=False)
algorithm_performance()
thompson_dict = {'reward':total_reward, 

                 'regret_list':regret_list, 

                 'ads_count':pd.Series(index_list).value_counts(normalize=True)}
regret = 0 

total_reward = 0

regret_list = [] 

index_list = [] 

impressions = [0,0] 

clicks = [0,0]

ctr = {0: [], 1: []}

total_reward = 0



for i in range(n):

    

    index = 0

    max_upper_bound = 0

    for k in [0,1]:

        if (impressions[k] > 0):

            CTR = clicks[k] / impressions[k]

            delta = math.sqrt(2 * math.log(i+1) / impressions[k])

            upper_bound = CTR + delta

            ctr[k].append(CTR)

        else:

            upper_bound = 1e400

        if upper_bound > max_upper_bound:

            max_upper_bound = upper_bound

            index = k

    index_list.append(index)

    impressions[index] += 1

    reward = bernoulli.rvs(ACTUAL_CTR[index])

    

    clicks[index] += reward

    total_reward += reward

    

    regret += max(ACTUAL_CTR) - ACTUAL_CTR[index]

    regret_list.append(regret)
algorithm_performance()
ucb1_dict = {'reward':total_reward, 

             'regret_list':regret_list, 

             'ads_count':pd.Series(index_list).value_counts(normalize=True)}
data1 = go.Bar(x=['Random Selection', 'Epsilon Greedy', 'Thompson Sampling', 'UCB1'],

               y=[random_dict['ads_count'][0], 

                  epsilon_dict['ads_count'][0], 

                  thompson_dict['ads_count'][0],

                  ucb1_dict['ads_count'][0]],

               name='Ad #0',

               marker=dict(color='rgba(10, 108, 94, .7)'))



data2 = go.Bar(x=['Random Selection', 'Epsilon Greedy', 'Thompson Sampling', 'UCB1'],

               y=[random_dict['ads_count'][1], 

                  epsilon_dict['ads_count'][1], 

                  thompson_dict['ads_count'][1],

                  ucb1_dict['ads_count'][1]],

               name='Ad #1',

               marker=dict(color='rgba(187, 121, 24, .7)'))



data = [data1, data2]

layout = go.Layout(title='Ratio of appearance of both Ads throughout the trials',

                   xaxis={'title': 'Algorithm'},

                   yaxis={'title': 'Ratio'},

                   barmode='stack')



fig = go.Figure(data=data, layout=layout)

iplot(fig)
data1 = go.Scatter(

    x=np.arange (0, n, 1),

    y=random_dict['regret_list'],

    name='Random Selection',

    marker=dict(color='#ffcc66')

)

data2 = go.Scatter(

    x=np.arange (0, n, 1),

    y=epsilon_dict['regret_list'],

    name='e-Greedy',

    marker=dict(color='#0099ff')

)

data3 = go.Scatter(

    x=np.arange (0, n, 1),

    y=thompson_dict['regret_list'],

    name='Thompson Sampling',

    marker=dict(color='#ff3300')

)

data4 = go.Scatter(

    x=np.arange (0, n, 1),

    y=ucb1_dict['regret_list'],

    name='UCB1',

    marker=dict(color='#33cc33')

)



layout = go.Layout(

    title='Regret by the Algorithm',

    xaxis={'title': 'Trial'},

    yaxis={'title': 'Regret'}

)



data = [data1, data2, data3, data4]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = go.Bar(

    x=[ucb1_dict['reward'], thompson_dict['reward'], epsilon_dict['reward'], random_dict['reward']],

    y=['UCB1', 'Thompson Sampling', 'e-Greedy','Random Selection'],

    orientation = 'h',

    marker=dict(color=['#33cc33', '#ff3300', '#0099ff', '#ffcc66']),

    opacity=0.7

)



text = go.Scatter(

    x=[ucb1_dict['reward'], thompson_dict['reward'], epsilon_dict['reward'], random_dict['reward']],

    y=['UCB1', 'Thompson Sampling', 'e-Greedy', 'Random Selection'],

    mode='text',

    text=[ucb1_dict['reward'], thompson_dict['reward'], epsilon_dict['reward'], random_dict['reward']],

    textposition='middle left',

    line = dict(

        color = ('rgba(255,141,41,0.6)'),

        width = 1

    ),

    textfont=dict(

        family='sans serif',

        size=16,

        color='#000000'

    )

)



data = [data,text]



layout = go.Layout(

    title='Total Reward by Algorithms',

    xaxis={'title': 'Total Reward (Clicks)'},

    margin={'l':200},

    showlegend=False

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)