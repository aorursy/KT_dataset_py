import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure



import time

from IPython.display import clear_output

from tqdm import tqdm



import warnings

warnings.filterwarnings("ignore")

sns.set(rc={'figure.figsize':(15, 10)})

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})



sns.set_style("whitegrid", {'axes.grid' : False})
states = ['A','B','C','D','E','F','G','T','M','recycling','trash']

x_list = [4,3,2,1,1,1,2,3,3,4,4]

y_list = [1,1,1,1,2,3,3,3,2,3,2]



# The low-level actions the agent can make in the environment

actions = ['left','right','up','down']

rewards = [-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,1,-1]



initial_policy = ['left','up','right','up','up','right','right','right','up','up','up']
for n,state in enumerate(states):

    if state == 'recycling':

        plt.scatter(x_list[n],y_list[n], s=150, color='g', marker='+')

    elif state == 'trash':

        plt.scatter(x_list[n],y_list[n], s=150, color='r', marker='x')

    else:

        plt.scatter(x_list[n],y_list[n], s=150, color='b')

    plt.text(x_list[n]+0.05,y_list[n]+0.05,states[n])

plt.title("Grid World Diagram for Classroom Paper Throwing Environment")

plt.xticks([])

plt.yticks([])

plt.ylim(0,4)

plt.xlim(0,5)

plt.show()

def action_outcome(state_x,state_y,action):

    if action == 'left':

        u = -1

        v = 0

    elif action == 'right':

        u = 1

        v = 0

    elif action == 'up':

        u = 0

        v = 1

    elif action == 'down':

        u = 0

        v = -1

    else:

        print("Error: Invalid action given")

        

    # Override if action hits wall to not move

    if (state_x == 1) & (u == -1):

        u = 0

        v = v

    elif (state_x == 4) & (u == 1):

        u = 0

        v = v

    elif (state_y == 1) & (v == -1):

        u = u

        v = 0

    elif (state_y == 3) & (v == 1):

        u = u

        v = 0

    elif (state_x == 2)&(state_y == 1) & (v == 1):

        u = u

        v = 0

    elif (state_x == 1)&(state_y == 2) & (u == 1):

        u = 0

        v = v  

    elif (state_x == 2)&(state_y == 3) & (v == -1):

        u = u

        v = 0         

    elif (state_x == 3)&(state_y == 2) & (u == -1):

        u = 0

        v = v 

    # Make so it cannot get out of bin

    elif (state_x == 4)&(state_y == 3):

        u = 0

        v = 0

    elif (state_x == 4)&(state_y == 2):

        u = 0

        v = 0

    return(u,v)
state_action_pairs = pd.DataFrame()

for n1,state in enumerate(states):

    action_list = pd.DataFrame()

    for n2,action in enumerate(actions):

        

        state_x = x_list[n1]

        state_y = y_list[n1]

        u = action_outcome(state_x,state_y,action)[0]

        v = action_outcome(state_x,state_y,action)[1]

        action_list  = action_list.append(pd.DataFrame({'state':state,'action':action,'x':x_list[n1],'y':y_list[n1],

                                                               'u':u,'v':v}, index=[(n1*len(actions)) + n2]))

    

    state_action_pairs = state_action_pairs.append(action_list)

state_action_pairs.head()
for n,state in enumerate(states):

    if state == 'recycling':

        plt.scatter(x_list[n],y_list[n], s=150, color='g', marker='+')

    elif state == 'trash':

        plt.scatter(x_list[n],y_list[n], s=150, color='r', marker='x')

    else:

        plt.scatter(x_list[n],y_list[n], s=150, color='b')

    plt.text(x_list[n]+0.05,y_list[n]+0.05,states[n])

    

    state_action_pairs_state = state_action_pairs[state_action_pairs['state']==state]

    plt.quiver([x_list[n],x_list[n],x_list[n],x_list[n]],[y_list[n],y_list[n],y_list[n],y_list[n]],

               state_action_pairs_state['u'],state_action_pairs_state['v'], alpha = 0.5)

    

plt.title("Grid World Diagram for Classroom Paper Throwing Environment: \n Possible Action Outcomes for each State \n (if action hits wall, next state is same as current)")

plt.xticks([])

plt.yticks([])

plt.ylim(0,4)

plt.xlim(0,5)

plt.show()

initial_policy_table = pd.DataFrame()



for n,state in enumerate(states):

    state_x = x_list[n]

    state_y = y_list[n]

    policy_a = initial_policy[n]

    

    u = action_outcome(state_x,state_y,policy_a)[0]

    v = action_outcome(state_x,state_y,policy_a)[1]

    

    initial_policy_table = initial_policy_table.append(pd.DataFrame({'state':state,'x':state_x,'y':state_y, 'action':policy_a,

                                                                       'u':u,'v':v}, index=[n]))

initial_policy_table
for n,state in enumerate(states):

    if state == 'recycling':

        plt.scatter(x_list[n],y_list[n], s=150, color='g', marker='+')

    elif state == 'trash':

        plt.scatter(x_list[n],y_list[n], s=150, color='r', marker='x')

    else:

        plt.scatter(x_list[n],y_list[n], s=150, color='b')

    plt.text(x_list[n]+0.05,y_list[n]+0.05,states[n])

    

    initial_policy_table_state = initial_policy_table[initial_policy_table['state']==state]

    plt.quiver([x_list[n],x_list[n],x_list[n],x_list[n]],[y_list[n],y_list[n],y_list[n],y_list[n]],

               initial_policy_table_state['u'],initial_policy_table_state['v'], alpha = 0.5)

    

plt.title("Grid World Diagram for Classroom Paper Throwing Environment: \n Initial Policy")

plt.xticks([])

plt.yticks([])

plt.ylim(0,4)

plt.xlim(0,5)

plt.show()

def environment(state, action):

    # Outcome probabilities

    if (state=='recycling')|(state=='trash'):

        prob = 0

    elif (state=='T'):

        prob = 1



    elif (state=='M'):

        prob = 0.7

   

    elif (state=='B'):

        prob = 0.7



    elif (state=='A'):

        prob = 0.7



    elif (state=='C'):

        prob = 0.7



    elif (state=='D'):

        prob = 0.7



    elif (state=='E'):

        prob = 0.7



    elif (state=='F'):

        prob = 0.7



    elif (state=='G'):

        prob = 0.7



    else:

        prob = "Error"

        print("Error state", state)



    action_rng = np.random.rand()

    if action_rng<=prob:

        action = action

    else:

        action_sub_list = actions.copy()

        action_sub_list.remove(action)

        action = random.choice(action_sub_list)

        

        

    state_x = x_list[states.index(state)]

    state_y = y_list[states.index(state)]

    u = action_outcome(state_x,state_y,action)[0]

    v = action_outcome(state_x,state_y,action)[1]

    next_state_x = state_x + u

    next_state_y = state_y + v

    # Returns index of x + y position to then find the state name

    next_state = states[' '.join(str(x_list[i])+ "_" + str(y_list[i]) for i in range(0,len(x_list))).split().index(str(next_state_x) + "_" + str(next_state_y))]

    reward = rewards[states.index(next_state)]

    return(state, action, state_x, state_y, u, v, next_state, next_state_x, next_state_y, reward)
for i in range(0,10):

    print(environment('A', 'left'))
import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure



import time

from IPython.display import clear_output

from tqdm import tqdm



import warnings

warnings.filterwarnings("ignore")



sns.set(rc={'figure.figsize':(15, 10)})

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5})



sns.set_style("whitegrid", {'axes.grid' : False})



#------------------------------------------------------------------------------------------



states = ['A','B','C','D','E','F','G','T','M','recycling','trash']

x_list = [4,3,2,1,1,1,2,3,3,4,4]

y_list = [1,1,1,1,2,3,3,3,2,3,2]



# The low-level actions the agent can make in the environment

actions = ['left','right','up','down']

rewards = [-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,1,-1]



initial_policy = ['left','up','right','up','up','right','right','right','up','up','up']



#------------------------------------------------------------------------------------------



def action_outcome(state_x,state_y,action):

    if action == 'left':

        u = -1

        v = 0

    elif action == 'right':

        u = 1

        v = 0

    elif action == 'up':

        u = 0

        v = 1

    elif action == 'down':

        u = 0

        v = -1

    else:

        print("Error: Invalid action given")

        

    # Override if action hits wall to not move

    if (state_x == 1) & (u == -1):

        u = 0

        v = v

    elif (state_x == 4) & (u == 1):

        u = 0

        v = v

    elif (state_y == 1) & (v == -1):

        u = u

        v = 0

    elif (state_y == 3) & (v == 1):

        u = u

        v = 0

    elif (state_x == 2)&(state_y == 1) & (v == 1):

        u = u

        v = 0

    elif (state_x == 1)&(state_y == 2) & (u == 1):

        u = 0

        v = v  

    elif (state_x == 2)&(state_y == 3) & (v == -1):

        u = u

        v = 0         

    elif (state_x == 3)&(state_y == 2) & (u == -1):

        u = 0

        v = v 

    # Make so it cannot get out of bin

    elif (state_x == 4)&(state_y == 3):

        u = 0

        v = 0

    elif (state_x == 4)&(state_y == 2):

        u = 0

        v = 0

    return(u,v)



def environment(state, action):

    # Outcome probabilities

    if (state=='recycling')|(state=='trash'):

        prob = 0

    elif (state=='T'):

        prob = 1



    elif (state=='M'):

        prob = 0.7

   

    elif (state=='B'):

        prob = 0.7



    elif (state=='A'):

        prob = 0.7



    elif (state=='C'):

        prob = 0.7



    elif (state=='D'):

        prob = 0.7



    elif (state=='E'):

        prob = 0.7



    elif (state=='F'):

        prob = 0.7



    elif (state=='G'):

        prob = 0.7



    else:

        prob = "Error"

        print("Error state", state)



    action_rng = np.random.rand()

    if action_rng<=prob:

        action = action

    else:

        action_sub_list = actions.copy()

        action_sub_list.remove(action)

        action = random.choice(action_sub_list)

        

        

    state_x = x_list[states.index(state)]

    state_y = y_list[states.index(state)]

    u = action_outcome(state_x,state_y,action)[0]

    v = action_outcome(state_x,state_y,action)[1]

    next_state_x = state_x + u

    next_state_y = state_y + v

    # Returns index of x + y position to then find the state name

    next_state = states[' '.join(str(x_list[i])+ "_" + str(y_list[i]) for i in range(0,len(x_list))).split().index(str(next_state_x) + "_" + str(next_state_y))]

    reward = rewards[states.index(next_state)]

    return(state, action, state_x, state_y, u, v, next_state, next_state_x, next_state_y, reward)



#------------------------------------------------------------------------------------------
