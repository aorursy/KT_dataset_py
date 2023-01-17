import gym # openAi gym

from gym import envs

import numpy as np
def seq(start, stop, step=1):

    n = int(round((stop - start)/float(step)))

    if n > 1:

        return([start + step*i for i in range(n+1)])

    elif n == 1:

        return([start])

    else:

        return([])
def train(gamma,alpha,epsilon):

    best_recompensa_media = -1

#     TROCAR PRA 5000

    for episodio in range(1,5001):

        concluido = False

        obs = env.reset()



        while concluido != True:

            if np.random.rand(1) < epsilon:

                action = env.action_space.sample()

            else:

                action = np.argmax(Q[obs])



            obs2, recompensa, concluido, info = env.step(action)

            Q[obs,action] += alpha * (recompensa + gamma * np.max(Q[obs2]) - Q[obs,action])

            obs = obs2

#     TROCAR PRA 500

        if episodio % 500 == 0:

            recompensa_media = 0.

            for i in range(100):

                obs= env.reset()

                concluido = False

                while concluido != True: 

                    action = np.argmax(Q[obs])

                    obs, recompensa, concluido, info = env.step(action)

                    recompensa_media += recompensa

            recompensa_media = recompensa_media/100



            if recompensa_media > best_recompensa_media:

                best_recompensa_media = recompensa_media

                best_episodio = episodio

#                 print("BEST: {} - {}".format(episodio, recompensa_media))

                

        

    return best_recompensa_media,best_episodio

     
env = gym.make('FrozenLake-v0')

env.reset()

NUM_ACTIONS = env.action_space.n

NUM_STATES = env.observation_space.n

Q = np.zeros([NUM_STATES, NUM_ACTIONS])



# gamma = 0.95 # fator de desconto

# alpha = 0.01 # taxa de aprendizagem

# epsilon = 0.1 # taxa de exploration

# best_recompensa_media = -1

table_best = [["GAMMA","ALPHA","EPSILON","RECOMPENSA"]]



for gamma in seq(0.90,0.98,0.01):

    for alpha in seq(0.005,0.02,0.005):

        for epsilon in seq(0.05,0.2,0.01):

            recompensa_media, episodio = train(gamma,alpha,epsilon)

            # if recompensa_media > best_recompensa_media:                

            best_recompensa_media = recompensa_media

            table_best.append([gamma,alpha,epsilon,best_recompensa_media])

            print('Gamma {}, Alpha {}, Epsilon {} - Episodio {} - Recompensa media: {}'.format(gamma,alpha,epsilon,episodio, recompensa_media))
import pandas as pd

dataframe=pd.DataFrame(table_best[1:], columns=table_best[0]) 

large5 = dataframe.nlargest(10, "RECOMPENSA") 

large5 
env = gym.make('Taxi-v2')



NUM_ACTIONS = env.action_space.n

NUM_STATES = env.observation_space.n



Q = np.zeros([NUM_STATES, NUM_ACTIONS])



gamma = 0.95 # fator de desconto

alpha = 0.90 # taxa de aprendizagem

epsilon = 0.1 # taxa de exploration



# MUDAR PARA 10001

for episodio in range(1,100001):

    

    concluido = False

    recompensa_total = 0

    obs = env.reset()

    

    while concluido != True:   

        

        if np.random.rand(1) < epsilon:

            action = env.action_space.sample()

        else:

            action = np.argmax(Q[obs])

            

        obs2, recompensa, concluido, info = env.step(action) # toma a acao

        Q[obs,action] += alpha * (recompensa + gamma * np.max(Q[obs2]) - Q[obs,action]) # atualiza a taxa da recompensa

        recompensa_total = recompensa_total + recompensa

        obs = obs2

        

    if episodio % 1000 == 0:

        recompensa_media = 0.

        for i in range(100):

            obs= env.reset()

            concluido = False

            while concluido != True: 

                action = np.argmax(Q[obs])

                obs, recompensa, concluido, info = env.step(action)

                recompensa_media += recompensa

        recompensa_media = recompensa_media/100

        print('Episodio {} - Recompensa media: {}'.format(episodio, recompensa_media))



        if recompensa_media > 9.1:

            print("Taxi solved")

            break
recompensa_total = 0

obs = env.reset()

env.render()

concluido = False



while concluido != True:

    action = np.argmax(Q[obs])

    obs, recompensa, concluido, info = env.step(action)

    recompensa_total = recompensa_total + recompensa

    env.render()

    

print("Recompensa: %r" % recompensa_total)