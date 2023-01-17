import numpy as np

import matplotlib.pyplot as plt

world_size = 5

discount = 0.9

reward_A = 10

reward_B = 5

reward_norm = 0

reward_offgrid = -1

# actions

actions = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])] # left, up, right, down

action_prob = .25

# special states

a_pos = [0, 1]

a_prime = [4, 1]

b_pos = [0, 3]

b_prime = [2, 3]



# There should not be any forced knowledge for the agent, just let it learn.

def next_state(state, action):

    '''

    >>> next_state([0, 1], np.array([0, -1]))

    ([4, 1], 10)

    '''

    if state == a_pos:

        n_state = a_prime

        reward = reward_A

    elif state == b_pos:

        n_state = b_prime

        reward = reward_B

    else:

        n_state = (np.array(state) + action).tolist()

        if (n_state[0] < 0) or (n_state[0] >= world_size) or (n_state[1] < 0) or (n_state[1] >= world_size):

            n_state = state # remained

            reward = reward_offgrid

        else:

            reward = reward_norm

    return n_state, reward



# doc test, pytest

import doctest

doctest.testmod(name='next_state', verbose=False)



def main():

    v = np.zeros((world_size, world_size))

    gridlist = {}

    num_iter = 100

    for i in range(num_iter):

        new_v = np.zeros_like(v)

        new_policy = np.zeros_like(v)

        for j in range(world_size):

            for k in range(world_size):

                gridlist[str(j)+'-'+str(k)]=[]

                for a in actions: # => not choosing max action but doing it for every action.

                    # take action

                    (x, y), reward = next_state([j, k], a)

                    # update v* 

                    temp = (reward + discount*v[x, y])

                    if (temp>=new_v[j, k]):

                        if (temp>new_v[j, k]):

                            gridlist[str(j)+'-'+str(k)]=[]

                        new_v[j, k] = temp if temp>new_v[j, k] else new_v[j, k]

                        gridlist[str(j)+'-'+str(k)].append(a)

        v = new_v

        

    plt.gcf().set_size_inches(10, 10)

    plt.xlim(0, world_size)

    plt.ylim(0, world_size)

    

    for i in range(world_size):

        for j in range(world_size):

            print("{:12.4f}".format(v[i, j]), end=' ')

            for a in gridlist[str(i)+'-'+str(j)]:

                plt.arrow(j+0.5,world_size-1-i+0.5,a[1]*0.3,-a[0]*0.3,head_width=0.05, head_length=0.1, fc='k', ec='k')

        print('\n')

    

    plt.grid(True)



    plt.show()

main()
