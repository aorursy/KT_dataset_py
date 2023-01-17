import numpy as np

import pandas as pd

import time

from IPython import display

from IPython.display import Image

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def init_q_table():

    

    #### Start ####  (≈ 2 code lines)  

    actions = np.array(['up', 'down', 'left', 'right'])

    q_table = pd.DataFrame(np.zeros((16, len(actions))), columns=actions)

    #### End ####

    

    return q_table
init_q_table()
def act_choose(state, q_table, epsilon):

    

    state_act = q_table.iloc[state,:]

    actions=np.array(['up','down','left','right'])

    

    #### Start #### (≈ 4 code lines)

    if (np.random.uniform() > epsilon or state_act.all() == 0):

        action = np.random.choice(actions)

    else:

        action = state_act.idxmax()

    #### End #### 

    return action
seed = np.random.RandomState(25) # In order to ensure the same result, the random number seed is introduced.

a = seed.rand(16, 4)

test_q_table = pd.DataFrame(a, columns=['up', 'down', 'left', 'right'])

l = []

for s in [1, 4, 7, 12, 14]:

    l.append(act_choose(state=s, q_table=test_q_table, epsilon=1))

l
def env_feedback(state, action,hole,terminal):

    reward = 0.

    end = 0

    a, b = state

    if action == 'up':

        a -= 1

        if a < 0:a = 0

        next_state = (a, b)  

    elif action == 'down':

        a += 1

        if a >= 4:a = 3

        next_state = (a, b)

    elif action == 'left':

        b -= 1

        if b < 0:b = 0

        next_state = (a, b)

    elif action == 'right':

        b += 1

        if b >= 4:b = 3

        next_state = (a, b)

        

    if next_state == terminal:

        reward = 10.

        end = 2

    elif next_state == hole:

        reward = -10.

        end = 1

    else:

        reward = -1.

        

    return next_state, reward, end
def update_q_table(q_table,state,action,next_state,next_action,terminal,gamma,alpha,reward):

    x, y = state

    next_x, next_y = next_state

    q_original= q_table.loc[x * 4 + y, action]

    

    if next_state != terminal:

        ### Start ### (≈ 1 code line)

        q_predict = reward + gamma * q_table.iloc[next_x * 4 + next_y].max()

        ### End ###

    else:

        q_predict = reward

        

    #### Start ### (≈ 1 code line)   

    q_table.loc[x * 4 + y, action] = (1-alpha) * q_original+alpha*q_predict

    ### End ###

    

    return q_table
new_q_table = update_q_table(q_table=test_q_table,state=(2,2),action='right',

                             next_state=(2,3),next_action='down',terminal=(3,2),

                             gamma =0.9 ,alpha =0.8 ,reward=10)
new_q_table.loc[10,'right']
def show_state(end, state, episode, step, q_table):

    terminal = (3, 2)

    hole = (2, 1)

    env = np.array([['_ '] * 4] * 4)

    env[terminal] = '$ '

    env[hole] = '# '

    env[state] = 'L '

    interaction = ''

    for i in env:

        interaction += ''.join(i) + '\n'

    if state == terminal:

        message = 'EPISODE: {}, STEP: {}'.format(episode, step)

        interaction += message

        display.clear_output(wait=True)

        print(interaction)

        print("\n"+"q_table:")

        print(q_table)

        time.sleep(3)  # Wait 3 seconds when you reach the end successfully

    else:

        display.clear_output(wait=True)

        print(interaction)

        print("\n"+"q_table:")

        print(q_table)

        time.sleep(0.3)  # Set the time consumption for each step
Image("/kaggle/input/week9dataset/maze_problem_with_sarsa.png")
def sarsa(max_episodes, alpha, gamma, epsilon):

    q_table = init_q_table()

    terminal = (3, 2)

    hole = (2, 1)

    episodes = 0

    while(episodes < max_episodes):

        step = 0

        state = (0, 0)

        end = 0

        show_state(end, state, episodes, step, q_table)

        x, y = state



        # Start ### （≈ 1 code line)

        next_action = act_choose(x * 4 + y, q_table, epsilon)  # Action choice

        ### End ###

        while(end == 0):

            x, y = state

            action = act_choose(x * 4 + y, q_table, epsilon)  # Action choice

            next_state, reward, end = env_feedback(state, action, hole, terminal)  # Environment rewards



            # Start ### （≈ 3 code lines)

            q_table = update_q_table(

                q_table, state, action, next_state,next_action, terminal, gamma, alpha, reward)  # Update q-table

            state = next_state

            action = None

            ### End ###

            step += 1

            show_state(end, state, episodes, step, q_table)

        if end == 2:

            episodes += 1
sarsa(max_episodes = 20,alpha = 0.8,gamma = 0.9,epsilon = 0.9)