# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gym  

import numpy as np

from collections import deque

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

import random

import matplotlib.pyplot as plt
env = gym.make("Taxi-v3").env 

env.render() 
print('State Space:',env.observation_space)
print('Action Space:',env.action_space) #Aksiyon sayısı
State = env.encode(3,1,2,3) # Taxi row, taxi column, passenger index, destination  

print(State)
env.s = State # to show the state 331 

env.render()
env.reset()



time_step = 0 #Passed time in while loop 

total_reward = 0 

list_visualize = []

while True: #To recognize the environment with a while loop

  time_step +=1

  #Choose Action

  action = env.action_space.sample() 

  #Perform Action & Get reward

  state, reward, done, _ = env.step(action) 



  #Total Reward

  total_reward +=reward

  list_visualize.append({"frame":env.render(),

                         "state":state,"action":action,

                         "reward":reward,"total_reward":total_reward

  

  

  })





  env.render()

  #Visualization



  if done:

     break
# Q table

q_table = np.zeros([env.observation_space.n,env.action_space.n])



#Hyper Parametrelerinin oluşturulması

alpha = 0.1

gamma = 0.9

epsilon = 0.1



#Plotting Metrix



reward_list = []

droputs_list = []



#Episode 



episode_number = 20000

for i in range(1,episode_number):



  #Inıtilaze Enviroment

  state = env.reset() 

    

  reward_count = 0 

  dropouts = 0





  while True:



    #Exploit & Explore to find action

    #epsilon=0.1 => % 10 explore %90 exploit

    if random.uniform(0,1) < epsilon:  

      action = env.action_space.sample()  

    else:

      action = np.argmax(q_table[state]) 

      



    #Action process and take reward / observation

    next_state, reward, done, _ = env.step(action) #step metodu actionı gerçekleştiren metottur. Bu 4 değişken döndürür. _ infodur ve kullanılmayacağı için bu şekilde tanımlanmıştır. 

    



    #Q learning function

    old_value = q_table[state,action] 

    next_max = np.max(q_table[next_state]) 

    next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max) 



    #Q table update

    q_table[state,action] = next_value 

 





    #Update State

    state = next_state # State'i next state eşitledik çünkü bir sonraki adımda next state state olacak.







    #find wrong drouputs

    if reward == -10:

      dropouts += 1



    reward_count += reward 



    if done:

      break



    if i%10 == 0:



      droputs_list.append(dropouts)

      reward_list.append(reward_count)

      print("Episode: {}, reward {}, wrong dropout {}".format(i,reward_count,dropouts))



    
fig ,axs = plt.subplots(1,2)

axs[0].plot(reward_list)

axs[0].set_xlabel("episode")

axs[0].set_ylabel("reward")



axs[1].plot(droputs_list)

axs[1].set_xlabel("episode")

axs[1].set_ylabel("dropouts")



axs[0].grid(True)

axs[1].grid(True)



plt.show()