import seaborn as sns

import matplotlib.pyplot as plt


bin_x = 0

bin_y = 0



starting_position_x = -5

starting_position_y = -5



plt.scatter(bin_x, bin_y, label = "Bin")

plt.scatter(starting_position_x, starting_position_y, label = "A")

plt.ylim([-10,10])

plt.xlim([-10,10])

plt.grid()

plt.legend()

plt.show()
import pandas as pd

import numpy as np


def probability(bin_x, bin_y, state_x, state_y, throw_deg):





    #First throw exception rule if person is directly on top of bin:

    if((state_x==bin_x) & (state_y==bin_y)):

        probability = 1

    else:

        

        

        # To accomodate for going over the 0 degree line

        if((throw_deg>270) & (state_x<=bin_x) & (state_y<=bin_y)):

            throw_deg = throw_deg - 360

        elif((throw_deg<90) & (state_x>bin_x) & (state_y<bin_y)):

            throw_deg = 360 + throw_deg

        else:

            throw_deg = throw_deg

            

        # Calculate Euclidean distance

        distance = ((bin_x - state_x)**2 + (bin_y - state_y)**2)**0.5



        # max distance for bin will always be on of the 4 corner points:

        corner_x = [-10,-10,10,10]

        corner_y = [-10,10,-10,10]

        dist_table = pd.DataFrame()

        for corner in range(0,4):

            dist = pd.DataFrame({'distance':((bin_x - corner_x[corner])**2 + (bin_y - corner_y[corner])**2)**0.5}, index = [corner])

            dist_table = dist_table.append(dist)

        dist_table = dist_table.reset_index()

        dist_table = dist_table.sort_values('distance', ascending = False)

        max_dist = dist_table['distance'][0]

        

        distance_score = 1 - (distance/max_dist)





        # First if person is directly horizontal or vertical of bin:

        if((state_x==bin_x) & (state_y>bin_y)):

            direction = 180

        elif((state_x==bin_x) & (state_y<bin_y)):

             direction = 0

        

        elif((state_x>bin_x) & (state_y==bin_y)):

             direction = 270

        elif((state_x<bin_x) & (state_y==bin_y)):

             direction = 90

              

        # If person is north-east of bin:

        elif((state_x>bin_x) & (state_y>bin_y)):

            opp = abs(bin_x - state_x)

            adj = abs(bin_y - state_y)

            direction = 180 +  np.degrees(np.arctan(opp/adj))



        # If person is south-east of bin:

        elif((state_x>bin_x) & (state_y<bin_y)):

            opp = abs(bin_y - state_y)

            adj = abs(bin_x - state_x)

            direction = 270 +  np.degrees(np.arctan(opp/adj))



        # If person is south-west of bin:

        elif((state_x<bin_x) & (state_y<bin_y)):

            opp = abs(bin_x - state_x)

            adj = abs(bin_y - state_y)

            direction =  np.degrees(np.arctan(opp/adj))



        # If person is north-west of bin:

        elif((state_x<bin_x) & (state_y>bin_y)):

            opp = abs(bin_y - state_y)

            adj = abs(bin_x - state_x)

            direction = 90 +  np.degrees(np.arctan(opp/adj))



        direction_score = (45-abs(direction - throw_deg))/45

      

        probability = distance_score*direction_score

        if(probability>0):

            probability = probability

        else:

            probability = 0

        

    return(probability)

    

    

    

bin_x = 0

bin_y = 0



starting_position_x = -5

starting_position_y = -5



test_1 = probability(bin_x, bin_y, starting_position_x, starting_position_y, 50)

test_2 = probability(bin_x, bin_y, starting_position_x, starting_position_y, 60)
print("Probability of first throw at 50 degrees = ", np.round(test_1,4))

print("Probability of second throw at 60 degress = ", np.round(test_2,4))
bin_x = 0

bin_y = 0

throw_direction = 180



prob_table = pd.DataFrame()

for i in range(0,20):

    state_x = -10 + i

    for j in range(0,20):

        state_y = -10 + j

        prob = pd.DataFrame({'x':state_x,'y':state_y,'prob':probability(bin_x, bin_y, state_x, state_y, throw_direction)}, index = [0])

        prob_table = prob_table.append(prob)

prob_table = prob_table.reset_index()



plt.scatter(prob_table['x'], prob_table['y'], s=prob_table['prob']*400, alpha=0.5)

plt.ylim([-10,10])

plt.xlim([-10,10])

plt.grid()

plt.title("Probability of Landing Shot for a given Thrown Direction: \n " + str(throw_direction)+" degrees")

plt.show()



        
from plotly.offline import init_notebook_mode, iplot, plot

from IPython.display import display, HTML

import plotly

import plotly.plotly as py



init_notebook_mode(connected=True)

bin_x = 0

bin_y = 0



prob_table = pd.DataFrame()

for z in range(0,37):

    throw_direction = z*10

    for i in range(0,20):

        state_x = -10 + i

        for j in range(0,20):

            state_y = -10 + j

            prob = pd.DataFrame({'throw_dir':throw_direction,'x':state_x,'y':state_y,'prob':probability(bin_x, bin_y, state_x, state_y, throw_direction)}, index = [0])

            prob_table = prob_table.append(prob)

prob_table = prob_table.reset_index(drop=True)

        
prob_table.head()
# Create interactive animation to show change of throwing direction to state probabilities (https://www.philiposbornedata.com/2018/03/01/creating-interactive-animation-for-parameter-optimisation-using-plot-ly/)

# This code is only used for visual and can be ignored entirely otherwise

prob_table_2 = prob_table

prob_table_2['continent'] = 'Test'

prob_table_2['country'] = 'Test2'



prob_table_2.columns = ['year', 'lifeExp', 'gdpPercap', 'pop', 'continent', 'country']

prob_table_2 = prob_table_2[prob_table_2['pop']>=0]

prob_table_2['pop'] = prob_table_2['pop']*100000000

prob_table_2.head()

alpha = list(set(prob_table_2['year']))

alpha = np.round(alpha,1)

alpha = np.sort(alpha)#[::-1]

years = np.round([(alpha) for alpha in alpha],1)

years



dataset = prob_table_2



continents = []

for continent in dataset['continent']:

    if continent not in continents:

        continents.append(continent)

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# fill in most of layout

figure['layout']['title'] = "Probability of Throwing Paper for each State as Thrown Direction Varies <br> PhilipOsborneData.com"

figure['layout']['xaxis'] = {'range': [-10,10], 'title': 'x'}

figure['layout']['yaxis'] = {'range': [-10,10],'title': 'y', 'type': 'linear'}

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1952',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}



figure['layout']['autosize'] = False

figure['layout']['width'] = 750

figure['layout']['height'] = 750





figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Thrown Direction: ',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



# make data

year = 1.0

for continent in continents:

    dataset_by_year = dataset[np.round(dataset['year'],1) == np.round(year,1)]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]



    data_dict = {

        'x': list(dataset_by_year_and_cont['lifeExp']),

        'y': list(dataset_by_year_and_cont['gdpPercap']),

        'mode': 'markers',

        'text': list(dataset_by_year_and_cont['country']),

        'marker': {

            'sizemode': 'area',

            'sizeref': 200000,

            'size': list(dataset_by_year_and_cont['pop'])

        },

        'name': continent

    }

    figure['data'].append(data_dict)



# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for continent in continents:

        dataset_by_year = dataset[np.round(dataset['year'],1) == np.round(year,1)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]



        data_dict = {

            'x': list(dataset_by_year_and_cont['lifeExp']),

            'y': list(dataset_by_year_and_cont['gdpPercap']),

            'mode': 'markers',

            'text': list(dataset_by_year_and_cont['country']),

            'marker': {

                'sizemode': 'area',

                'sizeref': 200000,

                'size': list(dataset_by_year_and_cont['pop'])

            },

            'name': continent

        }

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': str(year),

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure['layout']['sliders'] = [sliders_dict]



# iplot to do in notebook or plot to open new tab

iplot(figure)

#plot(figure)
#Define Q(s,a) table by all possible states and THROW actions initialised to 0

Q_table = pd.DataFrame()

for z in range(0,360):

    throw_direction = int(z)

    for i in range(0,21):

        state_x = int(-10 + i)

        for j in range(0,21):

            state_y = int(-10 + j)

            reward = 0

            Q = pd.DataFrame({'throw_dir':throw_direction,'move_dir':"none",'state_x':state_x,'state_y':state_y,'Q':0, 'reward': reward}, index = [0])

            Q_table = Q_table.append(Q)

Q_table = Q_table.reset_index(drop=True)

print("Q table 1 initialised")

Q_table.head()
#Define Q(s,a) table by all possible states and MOVE actions initialised to 0



for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        for m in range(0,8):

            move_dir = int(m)

            

            # skip impossible moves starting with 4 corners then edges

            if((state_x==10)&(state_y==10)&(move_dir==0)):

                continue

            elif((state_x==10)&(state_y==10)&(move_dir==2)):

                continue

                

            elif((state_x==10)&(state_y==-10)&(move_dir==2)):

                continue

            elif((state_x==10)&(state_y==-10)&(move_dir==4)):

                continue

                

            elif((state_x==-10)&(state_y==-10)&(move_dir==4)):

                continue

            elif((state_x==-10)&(state_y==-10)&(move_dir==6)):

                continue

                

            elif((state_x==-10)&(state_y==10)&(move_dir==6)):

                continue

            elif((state_x==-10)&(state_y==10)&(move_dir==0)):

                continue

                

            elif((state_x==10) & (move_dir == 1)):

                continue

            elif((state_x==10) & (move_dir == 2)):

                continue

            elif((state_x==10) & (move_dir == 3)):

                continue

                 

            elif((state_x==-10) & (move_dir == 5)):

                continue

            elif((state_x==-10) & (move_dir == 6)):

                continue

            elif((state_x==-10) & (move_dir == 7)):

                continue

                 

            elif((state_y==10) & (move_dir == 1)):

                continue

            elif((state_y==10) & (move_dir == 0)):

                continue

            elif((state_y==10) & (move_dir == 7)):

                continue

                 

            elif((state_y==-10) & (move_dir == 3)):

                continue

            elif((state_y==-10) & (move_dir == 4)):

                continue

            elif((state_y==-10) & (move_dir == 5)):

                continue

                 

            else:

                reward = 0

                Q = pd.DataFrame({'throw_dir':"none",'move_dir':move_dir,'state_x':state_x,'state_y':state_y,'Q':0, 'reward': reward}, index = [0])

                Q_table = Q_table.append(Q)

Q_table = Q_table.reset_index(drop=True)

print("Q table 2 initialised")

Q_table.tail()
Q_table[(Q_table['state_x']==-10) &(Q_table['throw_dir']=="none")].head(5)
Q_table_VI = Q_table.copy()
Q_table_VI['V'] = 0
bin_x = 0

bin_y = 0



prob_list = pd.DataFrame()

for n,action in enumerate(Q_table_VI['throw_dir']):

    # Guarantee 100% probability if movement

    if(action == "none"):

        prob = 1

    # Calculate if thrown

    else:

        prob = probability(bin_x, bin_y, Q_table_VI['state_x'][n], Q_table_VI['state_y'][n], action)

    prob_list = prob_list.append(pd.DataFrame({'prob':prob}, index = [n] ))

prob_list = prob_list.reset_index(drop=True)

Q_table_VI['prob'] = prob_list['prob']
Q_table_VI.head(5)
Q_table_VI[ (Q_table_VI['state_x']==-1) & (Q_table_VI['state_y']==-1) & (Q_table_VI['throw_dir']==45)]
import time

from IPython.display import clear_output
input_table = Q_table_VI.copy()

gamma = 0.8

num_repeats = 5



start_time = time.time()



output_metric_table = pd.DataFrame()

# Repeat until converges

for repeats in range(0,num_repeats):

    clear_output(wait=True)

    state_sub_full = pd.DataFrame()

    

    

    output_metric_table = output_metric_table.append(pd.DataFrame({'mean_Q':input_table['Q'].mean(), 

                                                                   'sum_Q': input_table['Q'].sum(),

                                                                   'mean_V':input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].mean(),

                                                                   'sum_V': input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].sum()}, index = [repeats]))

    

    

    # Iterate over all states defined by max - min of x times by max - min of y

    for x in range(0,21):

        state_x = -10 + x

        for y in range(0,21):

            state_y = -10 + y



            state_sub = input_table[ (input_table['state_x']==state_x) & (input_table['state_y']==state_y)]

            Q_sub_list = pd.DataFrame()

            for n, action in state_sub.iterrows():

                # Move action update Q

                if(action['throw_dir'] == "none"):

                    move_direction = action['move_dir']

                    #Map this to actual direction and find V(s) for next state

                    if(move_direction == 0):

                        move_x = 0

                        move_y = 1

                    elif(move_direction == 1):

                        move_x = 1

                        move_y = 1

                    elif(move_direction == 2):

                        move_x = 1

                        move_y = 0

                    elif(move_direction == 3):

                        move_x = 1

                        move_y = -1

                    elif(move_direction == 4):

                        move_x = 0

                        move_y = -1

                    elif(move_direction == 5):

                        move_x = -1

                        move_y = -1

                    elif(move_direction == 6):

                        move_x = -1

                        move_y = 0

                    elif(move_direction == 7):

                        move_x = -1

                        move_y = 1

                    Q = 1*(action['reward'] + gamma*max(input_table[ (input_table['state_x']==int(state_x+move_x)) & (input_table['state_y']==int(state_y+move_y))]['V']) )

                # Throw update Q +1 if sucessful throw or -1 if failed

                else:

                    Q = (action['prob']*(action['reward'] + gamma*1)) +  ((1-action['prob'])*(action['reward'] + gamma*-1))

                Q_sub_list = Q_sub_list.append(pd.DataFrame({'Q':Q}, index = [n]))

            state_sub['Q'] = Q_sub_list['Q']

            state_sub['V'] = max(state_sub['Q'])

            state_sub_full = state_sub_full.append(state_sub)

            

            



    

            

    input_table = state_sub_full.copy()

    print("Repeats completed: ", np.round((repeats+1)/num_repeats,2)*100, "%")

    

end_time = time.time()



print("total time taken this loop: ", np.round((end_time - start_time)/60,2), " minutes")
state_sub_full.head(3)
state_sub_full[ (state_sub_full['state_x']==-4) & (state_sub_full['state_y']==-4) & (state_sub_full['Q']== max(state_sub_full[ (state_sub_full['state_x']==-4) & (state_sub_full['state_y']==-4)]['Q']))]
output_metric_table
plt.plot(range(0,len(output_metric_table)), output_metric_table['mean_V'])

plt.title("Mean Q for all State-Action Pairs for each Update ")

plt.show()
Q_table_VI_3 = state_sub_full.copy()
optimal_action_list = pd.DataFrame()

for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        

        Q_table_VI_3

        

        optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                      'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][0],

                                      'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                     index = [state_y])

        optimal_action_list = optimal_action_list.append(optimal_action)

optimal_action_list = optimal_action_list.reset_index(drop=True)
optimal_action_list.head(5)
optimal_action_list[(optimal_action_list['state_x']==-1)&(optimal_action_list['state_y']==-1)]
optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )
sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

plt.title("Optimal Policy for Given Probabilities")

plt.ylim([-10,10])

plt.xlim([-10,10])

plt.show()
optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])

optimal_action_list.head(10)
arrow_scale = 0.1
# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))

optimal_action_list.head(5)
# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))

optimal_action_list.head(5)
x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values

plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

plt.title("Optimal Policy for Given Probabilities")

plt.show()

# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



Q_table_VI_3 = state_sub_full.copy()



optimal_action_list = pd.DataFrame()

for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        

        Q_table_VI_3

        

        optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                      'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][0],

                                      'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                     index = [state_y])

        optimal_action_list = optimal_action_list.append(optimal_action)

optimal_action_list = optimal_action_list.reset_index(drop=True)



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))

                                             

# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))

                                             

x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values



#plt.figure(figsize=(10, 10))

#plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

#sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

#plt.title("Optimal Policy for Given Probabilities")

#plt.show()



input_table = Q_table_VI.copy()

gamma = 0.8

num_repeats = 15



start_time = time.time()



output_metric_table = pd.DataFrame()

# Repeat until converges

for repeats in range(0,num_repeats):

    clear_output(wait=True)

    state_sub_full = pd.DataFrame()

    

    

    output_metric_table = output_metric_table.append(pd.DataFrame({'mean_Q':input_table['Q'].mean(), 

                                                                   'sum_Q': input_table['Q'].sum(),

                                                                   'mean_V':input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].mean(),

                                                                   'sum_V': input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].sum()}, index = [repeats]))

    

    

    # Iterate over all states defined by max - min of x times by max - min of y

    for x in range(0,21):

        state_x = -10 + x

        for y in range(0,21):

            state_y = -10 + y



            state_sub = input_table[ (input_table['state_x']==state_x) & (input_table['state_y']==state_y)]

            Q_sub_list = pd.DataFrame()

            for n, action in state_sub.iterrows():

                # Move action update Q

                if(action['throw_dir'] == "none"):

                    move_direction = action['move_dir']

                    #Map this to actual direction and find V(s) for next state

                    if(move_direction == 0):

                        move_x = 0

                        move_y = 1

                    elif(move_direction == 1):

                        move_x = 1

                        move_y = 1

                    elif(move_direction == 2):

                        move_x = 1

                        move_y = 0

                    elif(move_direction == 3):

                        move_x = 1

                        move_y = -1

                    elif(move_direction == 4):

                        move_x = 0

                        move_y = -1

                    elif(move_direction == 5):

                        move_x = -1

                        move_y = -1

                    elif(move_direction == 6):

                        move_x = -1

                        move_y = 0

                    elif(move_direction == 7):

                        move_x = -1

                        move_y = 1

                    Q = 1*(action['reward'] + gamma*max(input_table[ (input_table['state_x']==int(state_x+move_x)) & (input_table['state_y']==int(state_y+move_y))]['V']) )

                # Throw update Q +1 if sucessful throw or -1 if failed

                else:

                    Q = (action['prob']*(action['reward'] + gamma*1)) +  ((1-action['prob'])*(action['reward'] + gamma*-1))

                Q_sub_list = Q_sub_list.append(pd.DataFrame({'Q':Q}, index = [n]))

            state_sub['Q'] = Q_sub_list['Q']

            state_sub['V'] = max(state_sub['Q'])

            state_sub_full = state_sub_full.append(state_sub)

            

    

    

    ###

    # Create Quiver plot showing current optimal policy in one cell

    arrow_scale = 0.1



    Q_table_VI_3 = state_sub_full.copy()



    optimal_action_list = pd.DataFrame()

    for x in range(0,21):

        state_x = int(-10 + x)

        for y in range(0,21):

            state_y = int(-10 + y)



            Q_table_VI_3



            optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                          'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                          (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][0],

                                          'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                          (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                         index = [state_y])

            optimal_action_list = optimal_action_list.append(optimal_action)

    optimal_action_list = optimal_action_list.reset_index(drop=True)



    optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





    optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                             np.where(optimal_action_list['move_dir'] == 1, int(1),

                                             np.where(optimal_action_list['move_dir'] == 2, int(1),

                                             np.where(optimal_action_list['move_dir'] == 3, int(1),

                                             np.where(optimal_action_list['move_dir'] == 4, int(0),

                                             np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                             int(-1000)

                                            ))))))))

    optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                             np.where(optimal_action_list['move_dir'] == 1, int(1),

                                             np.where(optimal_action_list['move_dir'] == 2, int(0),

                                             np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 6, int(0),

                                             np.where(optimal_action_list['move_dir'] == 7, int(1),

                                             int(-1000)

                                            ))))))))

    optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



    # Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

    optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                        np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                        np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                        np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))



    # Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

    optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                        np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                        np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                        optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))







    x = optimal_action_list['state_x']

    y = optimal_action_list['state_y']

    u = optimal_action_list['u'].values

    v = optimal_action_list['v'].values

    plt.figure(figsize=(10, 10))

    plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

    sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

    plt.title("Optimal Policy for Given Probabilities for iteration " +str(repeats))

    #plt.savefig('E:\\Documents\\RL\\RL from scratch v2\\QuiverPlots\\'+str(repeats)+'.png')   # save the figure to file

    plt.close() 



    ###

    input_table = state_sub_full.copy()

    print("Repeats completed: ", np.round((repeats+1)/num_repeats,2)*100, "%")

    

end_time = time.time()



print("total time taken this loop: ", np.round((end_time - start_time)/60,2), " minutes")
plt.plot(range(0,len(output_metric_table)), output_metric_table['mean_V'])

plt.title("Mean Q for all State-Action Pairs for each Update ")

plt.show()
# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



Q_table_VI_3 = state_sub_full.copy()



optimal_action_list = pd.DataFrame()

for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        

        Q_table_VI_3

        

        optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                      'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][0],

                                      'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                     index = [state_y])

        optimal_action_list = optimal_action_list.append(optimal_action)

optimal_action_list = optimal_action_list.reset_index(drop=True)



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))



# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))



x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values

plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

plt.title("Optimal Policy for Given Probabilities")

plt.show()



optimal_action_list [ (optimal_action_list['state_x']==-10) & (optimal_action_list['state_y']==0)].head()
Q_table_VI_3 [ (Q_table_VI_3['state_x']==-5) & (Q_table_VI_3['state_y']==-5)].sort_values('Q', ascending=False).head()
# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



Q_table_VI_3 = state_sub_full.copy()



optimal_action_list = pd.DataFrame()

for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        

        Q_table_VI_3

        

        for i in range(0,len(Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'])):

            optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                          'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                          (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][i],

                                          'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                          (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                         index = [state_y])

            optimal_action_list = optimal_action_list.append(optimal_action)

optimal_action_list = optimal_action_list.reset_index(drop=True)



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))



# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))



x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values

plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action', alpha = 0.3)

plt.title("Optimal Policy for Given Probabilities")

plt.show()



input_table = Q_table_VI.copy()

gamma = 0.8

num_repeats = 10



start_time = time.time()



output_metric_table = pd.DataFrame()

# Repeat until converges

for repeats in range(0,num_repeats):

    clear_output(wait=True)

    state_sub_full = pd.DataFrame()

    

    

    output_metric_table = output_metric_table.append(pd.DataFrame({'mean_Q':input_table['Q'].mean(), 

                                                                   'sum_Q': input_table['Q'].sum(),

                                                                   'mean_V':input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].mean(),

                                                                   'sum_V': input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].sum()}, index = [repeats]))

    

    

    # Iterate over all states defined by max - min of x times by max - min of y

    for x in range(0,21):

        state_x = -10 + x

        for y in range(0,21):

            state_y = -10 + y



            state_sub = input_table[ (input_table['state_x']==state_x) & (input_table['state_y']==state_y)]

            Q_sub_list = pd.DataFrame()

            for n, action in state_sub.iterrows():

                # Move action update Q

                if(action['throw_dir'] == "none"):

                    move_direction = action['move_dir']

                    #Map this to actual direction and find V(s) for next state

                    if(move_direction == 0):

                        move_x = 0

                        move_y = 1

                    elif(move_direction == 1):

                        move_x = 1

                        move_y = 1

                    elif(move_direction == 2):

                        move_x = 1

                        move_y = 0

                    elif(move_direction == 3):

                        move_x = 1

                        move_y = -1

                    elif(move_direction == 4):

                        move_x = 0

                        move_y = -1

                    elif(move_direction == 5):

                        move_x = -1

                        move_y = -1

                    elif(move_direction == 6):

                        move_x = -1

                        move_y = 0

                    elif(move_direction == 7):

                        move_x = -1

                        move_y = 1

                    Q = 1*(action['reward'] + gamma*max(input_table[ (input_table['state_x']==int(state_x+move_x)) & (input_table['state_y']==int(state_y+move_y))]['V']) )

                # Throw update Q +1 if sucessful throw or -1 if failed

                else:

                    Q = (action['prob']*(action['reward'] + gamma*1)) +  ((1-action['prob'])*(action['reward'] + gamma*-1))

                Q_sub_list = Q_sub_list.append(pd.DataFrame({'Q':Q}, index = [n]))

            state_sub['Q'] = Q_sub_list['Q']

            state_sub['V'] = max(state_sub['Q'])

            state_sub_full = state_sub_full.append(state_sub)

            

    

    

    ###

    # Create Quiver plot showing current optimal policy in one cell

    arrow_scale = 0.1



    Q_table_VI_3 = state_sub_full.copy()



    optimal_action_list = pd.DataFrame()

    for x in range(0,21):

        state_x = int(-10 + x)

        for y in range(0,21):

            state_y = int(-10 + y)



            Q_table_VI_3



            for i in range(0,len(Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'])):

                optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                              'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                              (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][i],

                                              'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                              (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                             index = [state_y])

                optimal_action_list = optimal_action_list.append(optimal_action)

    optimal_action_list = optimal_action_list.reset_index(drop=True)



    optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





    optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                             np.where(optimal_action_list['move_dir'] == 1, int(1),

                                             np.where(optimal_action_list['move_dir'] == 2, int(1),

                                             np.where(optimal_action_list['move_dir'] == 3, int(1),

                                             np.where(optimal_action_list['move_dir'] == 4, int(0),

                                             np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                             int(-1000)

                                            ))))))))

    optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                             np.where(optimal_action_list['move_dir'] == 1, int(1),

                                             np.where(optimal_action_list['move_dir'] == 2, int(0),

                                             np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                             np.where(optimal_action_list['move_dir'] == 6, int(0),

                                             np.where(optimal_action_list['move_dir'] == 7, int(1),

                                             int(-1000)

                                            ))))))))

    optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



    # Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

    optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                        np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                        np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                        np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))



    # Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

    optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                        np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                        np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                        optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))







    x = optimal_action_list['state_x']

    y = optimal_action_list['state_y']

    u = optimal_action_list['u'].values

    v = optimal_action_list['v'].values

    plt.figure(figsize=(10, 10))

    plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

    sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action', alpha = 0.5)

    plt.title("Optimal Policy for Given Probabilities for iteration " +str(repeats))

    #plt.savefig('E:\\Documents\\RL\\RL from scratch v2\\QuiverPlots\\'+str(repeats)+'.png')   # save the figure to file

    plt.close() 



    ###

    input_table = state_sub_full.copy()

    print("Repeats completed: ", np.round((repeats+1)/num_repeats,2)*100, "%")

    

end_time = time.time()



print("total time taken this loop: ", np.round((end_time - start_time)/60,2), " minutes")
# SAVE OPTIMAL POLICY FOR LATER COMPARISON

# This is the .csv saved as the dataset and will be used when we apply RL algorithms in later outputs for comparison

Optimal_Policy_VI = optimal_action_list.copy()

#Optimal_Policy_VI.to_csv('E:\Documents\RL\RL from scratch v2\OptimalPolicy_angletol=45.csv')
input_table = Q_table_VI.copy()

input_table['reward'] = -0.05

gamma = 0.8

num_repeats = 10



start_time = time.time()



output_metric_table = pd.DataFrame()

# Repeat until converges

for repeats in range(0,num_repeats):

    clear_output(wait=True)

    state_sub_full = pd.DataFrame()

    

    

    output_metric_table = output_metric_table.append(pd.DataFrame({'mean_Q':input_table['Q'].mean(), 

                                                                   'sum_Q': input_table['Q'].sum(),

                                                                   'mean_V':input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].mean(),

                                                                   'sum_V': input_table[['state_x', 'state_y','V']].drop_duplicates(['state_x', 'state_y', 'V'])['V'].sum()}, index = [repeats]))

    

    

    # Iterate over all states defined by max - min of x times by max - min of y

    for x in range(0,21):

        state_x = -10 + x

        for y in range(0,21):

            state_y = -10 + y



            state_sub = input_table[ (input_table['state_x']==state_x) & (input_table['state_y']==state_y)]

            Q_sub_list = pd.DataFrame()

            for n, action in state_sub.iterrows():

                # Move action update Q

                if(action['throw_dir'] == "none"):

                    move_direction = action['move_dir']

                    #Map this to actual direction and find V(s) for next state

                    if(move_direction == 0):

                        move_x = 0

                        move_y = 1

                    elif(move_direction == 1):

                        move_x = 1

                        move_y = 1

                    elif(move_direction == 2):

                        move_x = 1

                        move_y = 0

                    elif(move_direction == 3):

                        move_x = 1

                        move_y = -1

                    elif(move_direction == 4):

                        move_x = 0

                        move_y = -1

                    elif(move_direction == 5):

                        move_x = -1

                        move_y = -1

                    elif(move_direction == 6):

                        move_x = -1

                        move_y = 0

                    elif(move_direction == 7):

                        move_x = -1

                        move_y = 1

                    Q = 1*(action['reward'] + gamma*max(input_table[ (input_table['state_x']==int(state_x+move_x)) & (input_table['state_y']==int(state_y+move_y))]['V']) )

                # Throw update Q +1 if sucessful throw or -1 if failed

                else:

                    Q = (action['prob']*(action['reward'] + gamma*1)) +  ((1-action['prob'])*(action['reward'] + gamma*-1))

                Q_sub_list = Q_sub_list.append(pd.DataFrame({'Q':Q}, index = [n]))

            state_sub['Q'] = Q_sub_list['Q']

            state_sub['V'] = max(state_sub['Q'])

            state_sub_full = state_sub_full.append(state_sub)

            

    input_table = state_sub_full.copy()

    print("Repeats completed: ", np.round((repeats+1)/num_repeats,2)*100, "%")

    

end_time = time.time()



print("total time taken this loop: ", np.round((end_time - start_time)/60,2), " minutes")
plt.plot(range(0,len(output_metric_table)), output_metric_table['mean_V'])

plt.title("Mean Q for all State-Action Pairs for each Update ")

plt.show()
# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



Q_table_VI_3 = state_sub_full.copy()



optimal_action_list = pd.DataFrame()

for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        

        Q_table_VI_3

        

        for i in range(0,len(Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                      (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'])):

            optimal_action = pd.DataFrame({'state_x':state_x, 'state_y': state_y, 

                                          'move_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                          (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['move_dir'][i],

                                          'throw_dir': Q_table_VI_3[ (Q_table_VI_3['state_x']==state_x) & (Q_table_VI_3['state_y']==state_y) &  (Q_table_VI_3['Q'] == max(Q_table_VI_3[(Q_table_VI_3['state_x']==state_x) & 

                                                          (Q_table_VI_3['state_y']==state_y)]['Q']))].reset_index(drop=True)['throw_dir'][0]},

                                         index = [state_y])

            optimal_action_list = optimal_action_list.append(optimal_action)

optimal_action_list = optimal_action_list.reset_index(drop=True)



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))



# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))



x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values

plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action', alpha = 0.5)

plt.title("Optimal Policy for Given Probabilities")

plt.show()


