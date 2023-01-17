import pandas as pd

import numpy as np

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from IPython.display import Image # Introduce the display module for convenience

import os
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods1.png")
Image("/kaggle/input/week9dataset/Quality_Based_Reinforcement_Learning_Methods2.jpeg")
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods3.png")
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods4.png")
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods5.png")
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods6.png")
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods7.png")
import time



def init_env():

    start=(0, 0)

    terminal=(3, 2)

    hole=(2, 1)

    env = np.array([['_ '] * 4] * 4) # Build a 4*4 environment 

    env[terminal] = '$ ' # Big ham

    env[hole] = '# ' # Trap

    env[start] = 'L '# Lion

    interaction = ''

    for i in env:

        interaction += ''.join(i) + '\n'

    print(interaction)
init_env()
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods8.png")
def init_q_table():

    actions = np.array(['up', 'down', 'left', 'right'])

    q_table = pd.DataFrame(np.zeros((16, len(actions))), columns=actions) 

    return q_table
init_q_table()
def act_choose(state, q_table, epsilon):

    """

    Parameters:

    state

    q_table

    epsilon -- The probability

    Returns:

    action -- Next action

    """

    state_act = q_table.iloc[state, :]

    actions = np.array(['up', 'down', 'left', 'right'])

    if (np.random.uniform() > epsilon or state_act.all() == 0):

        action = np.random.choice(actions)

    else:

        action = state_act.idxmax()

    return action
def env_feedback(state, action, hole, terminal):

    """

    Parameters:

    state

    action

    hole -- Where the trap is

    terminal

    Returns:

    next_state

    reward

    end -- The end signal

    """

    reward = 0.

    end = 0

    a, b = state

    if action == 'up':

        a -= 1

        if a < 0:

            a = 0

        next_state = (a, b)

    elif action == 'down':

        a += 1

        if a >= 4:

            a = 3

        next_state = (a, b)

    elif action == 'left':

        b -= 1

        if b < 0:

            b = 0

        next_state = (a, b)

    elif action == 'right':

        b += 1

        if b >= 4:

            b = 3

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
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods9.png")
def update_q_table(q_table, state, action, next_state, terminal, gamma, alpha, reward):

    """

    Parameters:

    q_table

    state

    action

    next_state

    terminal

    gamma -- The discount factor

    alpha -- The learning rate

    reward

    Returns:

    q_table -- Updated Q-Table

    """

    x, y = state

    next_x, next_y = next_state

    q_original = q_table.loc[x * 4 + y, action]

    if next_state != terminal:

        q_predict = reward + gamma * q_table.iloc[next_x * 4 + next_y].max()

    else:

        q_predict = reward

    q_table.loc[x * 4 + y, action] = (1-alpha) * q_original+alpha*q_predict

    return q_table
def show_state(end, state, episode, step, q_table):

    """

    Parameters:

    end -- End signal

    state 

    episode -- The number of iterations

    step -- The step of iteration

    q_table-- Q-Table

    """

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

        display.clear_output(wait=True)  # Clear

        print(interaction)

        print("\n"+"q_table:")

        print(q_table)

        time.sleep(3)  # Wait 3 seconds

    else:

        display.clear_output(wait=True)

        print(interaction)

        print(q_table)

        time.sleep(0.3)  # Control the time taken for each step
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods12.png")
def q_learning(max_episodes, alpha, gamma, epsilon):

    """

    Parameters:

    max_episodes -- The maximum number of iterations

    alpha -- Learning rate

    gamma -- Discount factor

    epsilon -- Probability

    Returns:

    q_table -- Updated Q-Table

    """

    q_table = init_q_table()

    terminal = (3, 2)

    hole = (2, 1)

    episodes = 0

    while(episodes <= max_episodes):

        step = 0

        state = (0, 0)

        end = 0

        show_state(end, state, episodes, step, q_table)

        while(end == 0):

            x, y = state

            act = act_choose(x * 4 + y, q_table, epsilon)  # Choose action

            next_state, reward, end = env_feedback(

                state, act, hole, terminal)  # Reward from the environment

            q_table = update_q_table(

                q_table, state, act, next_state, terminal, gamma, alpha, reward)  # q-table update

            state = next_state

            step += 1

            show_state(end, state, episodes, step, q_table)

        if end == 2:

            episodes += 1
from IPython import display

q_learning(max_episodes=10, alpha=0.8, gamma=0.9, epsilon=0.9)
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods12.png")
Image("/kaggle/input/week9dataset/Quality-based Reinforcement Learning Methods13.png")