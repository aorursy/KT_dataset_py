!pip install kaggle-environments



import os

import time

import numpy as np

import time

import random

from collections import defaultdict



import tensorflow as tf

from tensorflow.keras import Model,models,layers



from kaggle_environments import make

from kaggle_environments.envs.halite.helpers import *



import matplotlib.pyplot as plt

import pandas as pd



pd.set_option('display.max_columns', None)

np.set_printoptions(precision=2)
BOARD_SIZE = 21

b2 = BOARD_SIZE//2

N_ACTIONS = 5

N_AGENTS = 4

FEATURE_SIZE = 1+3*N_AGENTS

INIT_ENV = "1yard1ship"



# you can experiment with these

DISCOUNT = 0.90

HNORM = 1000

IDLE_COST = 2

N_RTG = 6   # reward-to-go steps



SAVE_DIR = "/checkpoints/"



ship_action_encode = {0:ShipAction.NORTH,

                 1:ShipAction.SOUTH,

                 2:ShipAction.WEST,

                 3:ShipAction.EAST,

                 4:None,

                 5:ShipAction.CONVERT} # convert is not used in this notebook



shipyard_action_encode = {0:None,

                          1:ShipyardAction.SPAWN}
def initializeEnv(option=None,ff=0):

    """

    creates a new environment

    

    option: init option, (e.g. 1yard1ship, converts first ship to shipyard and spawn a ship)

    

    ff: fast forward, waits ff steps before doing initialization moves

    """

    env = make("halite", configuration={"size": BOARD_SIZE,"episodeSteps":400})

    env.reset(N_AGENTS)    



    for i in range(ff): # fast forward

      _ = env.step(None)



    if option == "1yard1ship":   



      board = Board(env.state[0].observation, env.configuration) 

      for player in board.players.values():     

        player.ships[0].next_action = ShipAction.CONVERT

      state = env.step([player.next_actions for player in board.players.values()])[0]

      

      board = Board(env.state[0].observation, env.configuration) 

      for player in board.players.values():     

        player.shipyards[0].next_action = ShipyardAction.SPAWN

      state = env.step([player.next_actions for player in board.players.values()])[0]



    return env
def getObs(board,entity):

    """returns observation of an entity (ship or shipyard) of a player

    (BOARD_SIZE, BOARD_SIZE, FEATURE_SIZE)

    

    currently features:

    

    -halite on cells

    

    for each player (starting from the player the entity belong to)

        -player ships

        -players ships halite in cargo

        -player shipyards

    

    halite values are divided by HNORM constant

    """

    x,y = coordTransform(entity.position)



    b2 = BOARD_SIZE//2

    layers =[]



    halites = np.reshape(board.observation["halite"],(BOARD_SIZE,BOARD_SIZE))

    halites = np.roll(halites,(b2-x,b2-y),axis=(0,1)) 

    layers.append(halites/HNORM)



    for player in [board.current_player]+list(board.opponents):



      array_layer = np.zeros((BOARD_SIZE,BOARD_SIZE))

      array_layer2 = np.zeros((BOARD_SIZE,BOARD_SIZE))

      for ship in player.ships:

          i,j = coordTransform(ship.position)

          array_layer[i,j] = 1

          array_layer2[i,j] = ship.halite/HNORM

      array_layer = np.roll(array_layer,(b2-x,b2-y),axis=(0,1)) 

      array_layer2 = np.roll(array_layer2,(b2-x,b2-y),axis=(0,1))

      layers.append(array_layer)   

      layers.append(array_layer2)   



      array_layer = np.zeros((BOARD_SIZE,BOARD_SIZE))

      for shipyard in player.shipyards:

          i,j = coordTransform(shipyard.position)

          array_layer[i,j] = 1

      array_layer = np.roll(array_layer,(b2-x,b2-y),axis=(0,1)) 

      layers.append(array_layer)    

    

    layers = tf.convert_to_tensor(layers,dtype=tf.float32)

    layers = tf.transpose(layers, [1, 2, 0])

    return layers



def coordTransform(coords):

  """ change coordinates returned by (entity).position method to numpy coordinates

  """

  x,y = coords

  x_new = BOARD_SIZE-1-y

  y_new = x

  return x_new,y_new  
# exploration policies

def Boltzmann(model,X,T):

  if T == None:

    T = 0

  probs = tf.math.exp(probs/T)/tf.reduce_sum(tf.math.exp(probs/T))

  action = np.random.choice(N_ACTIONS, p=np.squeeze(probs))

  return action



def EGreedy(model,X,epsilon):

  if epsilon==None:

    epsilon = 0

  if np.random.rand()<epsilon:

    action = np.random.randint(model.output.shape[1])

  else:

    probs = model(X)

    action = tf.argmax(probs,axis=1)

  return action



class DoubleDQN:

  def __init__(self,model,target_update_method="periodic",

                          tau=0.999,

                          exploration="egreedy"):

    self.Qnet = model

    self.target_update_method = target_update_method

    self.tau = tau

    if exploration=="egreedy":

      self.explorationPolicy = EGreedy

    if exploration=="boltzmann":

      self.explorationPolicy = Boltzmann



  def compile(self,optimizer):



    def masked_error(args):

            y_true, y_pred, mask = args

            loss = huber_loss(y_true*mask, y_pred*mask)

            return loss



    self.Qnet.compile(loss="mse",optimizer="Adam") # this is not important

    self.targetNet = models.clone_model(self.Qnet)





    # these couple of lines adapted from https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py

    # Qnet predicts Q values for each action. Target Q values come from rewards of selected actions. 

    # So we need to mask losses for actions that are not selected.

    # We add a mask layer on top of Q net

    y_pred = self.Qnet.output

    y_true = layers.Input(name='y_true', shape=(self.Qnet.output.shape[1],))

    mask_layer = layers.Input(name='mask', shape=(self.Qnet.output.shape[1],))

    loss_out = layers.Lambda(masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask_layer])

    ins = [self.Qnet.input] if type(self.Qnet.input) is not list else self.Qnet.input

    self.Qnet_trainable = Model(inputs=ins + [y_true, mask_layer], outputs=[loss_out])

    masked_loss = lambda y_true, y_pred: y_pred

    self.Qnet_trainable.compile(optimizer=optimizer, loss=[masked_loss])

  



  def copy_to_target(self):

    self.targetNet = models.clone_model(self.Qnet)

    for t, e in zip(self.targetNet.trainable_variables, self.Qnet.trainable_variables):

              t.assign(e)



  def load_weights(self, filepath):

    self.Qnet.load_weights(filepath)

    self.copy_to_target()



  def save_weights(self, filepath, overwrite=True):

    self.Qnet.save_weights(filepath, overwrite=overwrite)



  def __call__(self,X,expl_param=None,model="Q"):

    if model=="T":

      q_values = self.targetNet(X)

      return q_values

    if model=="Q":

      actions = self.explorationPolicy(self.Qnet,X,expl_param)

      return actions



  def train_on_batch(self,X,targets,mask):

    loss = self.Qnet_trainable.train_on_batch([X,targets,mask],[targets])

    return loss



  def update_target(self):

    if self.target_update_method=="polyak":

      for t, e in zip(self.targetNet.trainable_variables, self.Qnet.trainable_variables):

        t.assign(t * self.tau + e * (1 - self.tau))



    if self.target_update_method=="periodic":

      self.copy_to_target()

      

              
# neural network



huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)



grid_input = layers.Input(shape=(BOARD_SIZE,BOARD_SIZE,FEATURE_SIZE)) #for features extracted from map



conv1 = layers.Conv2D(16, (1, 1), activation="tanh",

                     kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001))(grid_input)



conv2 = layers.Conv2D(16, (3, 3), activation="tanh",

                     kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001))(conv1)



flat = layers.Flatten()(conv2)





flat_input = layers.Input(shape=(1,)) # input for undeserved halite in cargo. Future global/scalar values will append here

merged = layers.Concatenate(axis=1)([flat, flat_input])



d1 = layers.Dense(32,activation="tanh",

                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001))(merged)



d2 = layers.Dense(N_ACTIONS,activation="tanh",

                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001))(d1)



qnet = models.Model(inputs=[grid_input, flat_input], outputs=d2)
dqn = DoubleDQN(qnet)

dqn.compile(optimizer="Adam")
class data_collector:

    def __init__(self,player_id,expl_param=1):

        self.player_id = player_id

        self.spawn_probability = 0.5

        self.expl_param = expl_param

        self.discount_factor = DISCOUNT



        self.transitions = {}  

        self.state0_ship_halite = {}

        self.state0_ships_loc = {}



        # to be reset after end of episodes

        self.prev_und_gains = defaultdict(lambda: 0)

        self.paths = defaultdict(list)



    def process_0(self,observation,configuration):

        # State 0

        self.transitions = {}  

        self.state0_ship_halite = {}  

        self.state0_ships_loc = {}



        board = Board(observation, configuration)

        me = board.players[self.player_id]

        for shipyard in me.shipyards:

          

          syard_obs = getObs(board,shipyard)

          loaded_ship_nearby = ((syard_obs[:,:,1][b2-1,b2] + syard_obs[:,:,1][b2+1,b2] +

                                syard_obs[:,:,1][b2,b2-1] + syard_obs[:,:,1][b2,b2+1]) > (10/HNORM))

                                # nearby: 1 manhattan distance

          if not loaded_ship_nearby:

            if np.random.rand()<self.spawn_probability:

              shipyard.next_action = shipyard_action_encode[1]



        for ship in me.ships:

            ID = ship.id

            self.transitions[ID] = []



            self.state0_ships_loc[ID] = coordTransform(ship.position)

            self.state0_ship_halite[ID] = ship.halite

            ship_obs = getObs(board,ship)           # obs 0

            ship_und_gain = self.prev_und_gains[ID]

            self.transitions[ID].append(ship_obs)

            self.transitions[ID].append(ship_und_gain)



            ship_obs = tf.expand_dims(ship_obs, 0) # batch dimension 

            ship_und_gain = tf.expand_dims(ship_und_gain, 0)



            action = dqn([ship_obs,ship_und_gain],self.expl_param)

            ship.next_action = ship_action_encode[int(action)]

            self.transitions[ID].append(int(action))

        return me.next_actions



    def process_1(self,observation,configuration):

        # State 1

        board = Board(observation, configuration)

        me = board.players[self.player_id]  

        state1_ships_loc = {ship.id: coordTransform(ship.position) for ship in me.ships}



        ships_destroyed = np.setdiff1d(list(self.state0_ships_loc),list(state1_ships_loc))

        team_killers = []

        undeserved_gain = {}

    

        for ID in ships_destroyed:

          ship_next_obs = tf.identity(self.transitions[ID][0])

          ship_next_obs = ship_next_obs.numpy()

          ship_next_obs[:,:,1][b2,b2]=0

          ship_next_obs = tf.convert_to_tensor(ship_next_obs)

          self.transitions[ID].append(ship_next_obs)



          # calculate where the victim is destroyed 

          x,y = self.state0_ships_loc[ID] 

          last_action = (int(self.transitions[ID][2]) == np.arange(5)) * np.array([-1,1,-1,1,0]) # 

          crimeX,crimeY = x+np.sum(last_action[:2]),y+np.sum(last_action[2:5])



          team_killerFound = False

        

          for IDsus in state1_ships_loc:                            # looking for guilty ship

            # ( It would be much easier to do these checks with built-in methods.- cell.ships etc.)

            if IDsus not in self.state0_ships_loc:                    # skip ships that spawned in state1

              continue

            if state1_ships_loc[IDsus] == (crimeX,crimeY):          # ally at crime location

              if (self.transitions[IDsus][2]==4):                      #  ally was standing. No punishment    

                undeserved_gain[IDsus] =  self.state0_ship_halite[ID] #  halite taken from victim         

              else:

                team_killers.append(IDsus)

                team_killerFound = True

                undeserved_gain[IDsus] =  self.state0_ship_halite[ID]

            

          if team_killerFound:

            self.transitions[ID].append(self.prev_und_gains[ID])

            reward = 0                      # no punishment

            self.transitions[ID].append(reward)  

            gamma = 0

            self.transitions[ID].append(gamma)      



          else: # killed by enemy | victim crashed suspect | new ship spawned on the location 

            self.transitions[ID].append(self.prev_und_gains[ID])

            reward = (-500/HNORM)           # punishing the victim

            self.transitions[ID].append(reward)       

            gamma = 0                       # this consequence only depend on last action (maybe not?)

            self.transitions[ID].append(gamma)



        # alive ships

        for ID in list(state1_ships_loc):

          if ID not in self.state0_ships_loc:       # skip ships that spawned in state1

            state1_ships_loc.pop(ID)

            continue

          ship = [ship for ship in me.ships if ID==ship.id][0]

          ship_next_obs = getObs(board,ship)  # obs 1

          self.transitions[ID].append(ship_next_obs) 



          ship_gain = (ship.halite - self.state0_ship_halite[ID])

          returned = int(ship_next_obs[:,:,3][b2,b2]) # if arrived on a friendly shipyard                  

          idle = (self.transitions[ID][2] == 4) and (ship_gain==0)               # if stayed on a cell without halite

          if ID not in undeserved_gain:

            undeserved_gain[ID] = 0

          self.prev_und_gains[ID] += undeserved_gain[ID]

          self.transitions[ID].append(self.prev_und_gains[ID])

          

          reward = (ship_gain

                    +2*returned*self.state0_ship_halite[ID]  # if returned, ship_gain + 2*(ships prev halite) = ships prev halite  

                    -returned*(self.prev_und_gains[ID])                 

                    -idle*IDLE_COST

                    -(ID in team_killers)*500

                    -undeserved_gain[ID])/HNORM        # ship_gain and undeserved_gain will cancel out

          

          

          self.transitions[ID].append(reward) 

          

          if returned:

            self.prev_und_gains[ID] = 0



          gamma = 0 if ID in team_killers else self.discount_factor

          self.transitions[ID].append(gamma)



        for ID in self.transitions:

          self.paths[ID].append(self.transitions[ID])    

    

def reward_to_go(rewards, gammas, N = N_RTG):

    """

    Future rewards affect current reward for N steps.

    N = None: All future rewards 

    """

    rlen = len(rewards)

    res = []

    if N == None:

      for i in range(rlen-1):

          res.append(rewards[i] + np.sum(rewards[i+1:]*(gammas[i+1:]**np.arange(1,rlen-i,dtype=np.float32))))

      res.append(rewards[-1])

    else:

      for i in range(rlen-1):

          res.append(rewards[i] + np.sum(rewards[i+1:i+N]*(gammas[i+1:i+N]**np.arange(1,min(N,rlen-i),dtype=np.float32))))

      res.append(rewards[-1])

    return res
def simulate(expl_param=0,initOption = INIT_ENV,steps=400,ff=0,rtg=False,spawn_probability=0.5):

  """  creates an env., runs an episode, returns rewards of each ship

  """

  env = initializeEnv(option=initOption)

  step = env.state[0]["observation"]["step"]

  board = Board(env.state[0].observation, env.configuration)

  agents = [data_collector(ID,expl_param) for ID in list(board.players)]



  rewards = defaultdict(lambda: np.full([steps], np.nan))

  gammas = defaultdict(list)

  belongTo = defaultdict(set)

  while not env.done:     

      actions = []

      for agent in agents:

        actions.append(agent.process_0(env.state[0].observation, env.configuration))

        

      env.step(actions)

    

      for agent in agents:

        agent.process_1(env.state[0].observation, env.configuration)

        for ID in agent.transitions:

          rewards[ID][step] = agent.transitions[ID][5]

          gammas[ID].append(agent.transitions[ID][6])

          belongTo[agent.player_id].add(ID)

      step += 1

  if rtg:

    for ID in rewards:

      rewards[ID][~np.isnan(rewards[ID])] = reward_to_go(rewards[ID][~np.isnan(rewards[ID])],gammas[ID],N=N_RTG)

  env.render(mode="ipython",width=800, height=600)

  return rewards,belongTo
def generateSamples(batch_size,expl_param):  

  samples = []

  env = initializeEnv(option="1yard1ship")

  board = Board(env.state[0].observation, env.configuration)

  agents = [data_collector(ID,expl_param) for ID in list(board.players)]



  while len(samples)<batch_size:

    while not env.done:

      actions = []

      board = Board(env.state[0].observation, env.configuration) 

        

      for agent in agents: # sample actions for each player

        actions.append(agent.process_0(env.state[0].observation, env.configuration))



      env.step(actions)    # update environment

      

      for agent in agents: # calculate rewards

        agent.process_1(env.state[0].observation, env.configuration)



    for agent in agents:    

      # recalculating rewards: R + (gamma*future_R)

      for p in agent.paths:

        rewards = [t[5] for t in agent.paths[p]]

        gammas = [t[6] for t in agent.paths[p]]

        rewards = reward_to_go(rewards,gammas,N=N_RTG)

        for i,t in enumerate(agent.paths[p]):

          t[5] = rewards[i]

          samples.append(t)    

      agent.paths = defaultdict(list)

      agent.prev_und_gains = defaultdict(lambda: 0)



    env = initializeEnv(option="1yard1ship")

  return samples
max_size_memory = 300000 # depends on your memory limitation. 

init_size_memory = 20000
memory = generateSamples(batch_size=init_size_memory,expl_param=1) # create random dataset
len(memory)
def mirrorAction(a,axis):

  # n,s,w,e = 0,1,2,3

  if a > 3:          # 4: "None"

    new_a = a

  else:

    if axis=="h": # w<>e

      new_a = 2*(a==3) + 3*(a==2) + 1*(a==1)



    if axis=="v": # n<>s

      new_a = 2*(a==2) + 3*(a==3) + (not a)

  

  return new_a



def augmentData(transitions):

  """ 

  mirroring transitions to increase data

  transition: (s,a,s',r,g)

  (s,a,s')  will be mirrored

  """



  length = len(transitions)

  for i in range(length):

    

    t = transitions[i].copy()



    t[0] = tf.reverse(t[0],(1,)) # horizontal (west,east change)

    t[3] = tf.reverse(t[3],(1,))

    t[2] = mirrorAction(t[2],"h")

    transitions.append(t)



    t = t.copy()

    t[0] = tf.reverse(t[0],(0,)) # vertical (north,south change)

    t[3] = tf.reverse(t[3],(0,))

    t[2] = mirrorAction(t[2],"v")

    transitions.append(t)



    t = t.copy()

    t[0] = tf.reverse(t[0],(1,)) # horizontal again 

    t[3] = tf.reverse(t[3],(1,))

    t[2] = mirrorAction(t[2],"h")

    transitions.append(t)
a_sample = memory[42]
# Check if layers (ships, halite, shipyards) moved correctly after action.

print("halites at state 0")

print(a_sample[0][5:16,5:16,0])       # printing 5:15 because 21x21 is too big for output cell 

print("Action: %s"%ship_action_encode[a_sample[2]])

print("halites at state 1")

print(a_sample[3][5:16,5:16,0])

print("Reward: %f"%a_sample[5])
a_sample = [a_sample]

augmentData(a_sample)
len(a_sample) # 3 new samples created from 1
# Checking the one that should also flip the action. e.g. Action NORTH should become SOUTH

print(a_sample[2][0][5:16,5:16,0]) # halites at state 0

print("Action: %s"%ship_action_encode[a_sample[2][2]])

print(a_sample[2][3][5:16,5:16,0]) # halites at state 1

print("Reward: %f"%a_sample[1][5])

print(len(memory))

augmentData(memory)

print(len(memory))
# arbitrary values

n_iter = 10000

batch_size = 1000

add_to_memory_every = 100

update_target_every = 200

save_model_every  = 400



itr_last = 0

losses = []
start = time.time()

# epsilon = 0.9 

for itr in range(itr_last,n_iter,1):

  epsilon = 1-(0.3*itr/n_iter)                          # decreasing epsilon gradually



  samples = random.sample(memory, batch_size)

  obs_batch, scalar_batch, actions_batch, next_obs_batch,next_scalar_batch,rewards_batch, gammas_batch = map(tf.stack,zip(*samples))



  q_values_target = dqn([next_obs_batch,next_scalar_batch],model="T") # next Q values from target network



  mask = np.zeros(shape=q_values_target.shape)    

  for i in range(mask.shape[0]):

    mask[i][actions_batch[i]]=1 



  rewards_batch = tf.cast(tf.expand_dims(rewards_batch,1),dtype=tf.float32) # fixing dimension

  targets_batch =  rewards_batch + DISCOUNT * q_values_target

  loss = dqn.train_on_batch([obs_batch,scalar_batch],targets_batch,mask)



  # generating new samples

  if itr%add_to_memory_every == 0:

    new_samples = generateSamples(batch_size=batch_size,expl_param=epsilon)

    augmentData(new_samples)

    memory.extend(new_samples)

    while len(memory) > max_size_memory:

          memory.pop(0)



  # updating target network

  if dqn.target_update_method == "periodic":         

    if itr%update_target_every==0:

      dqn.update_target()

  if dqn.target_update_method == "polyak":

    dqn.update_target()   



  losses.append(loss)

  itr_last = itr



  if itr%1000==0:

    print("---- Iteration %d ----"%itr)

    print("Epsilon: %f"%epsilon)

    print("Loss: %f"%loss)

    print(time.time()-start)

#   if itr%save_model_every == 0:

#     print("Saving model")

#     dqn.save_weights(SAVE_DIR + 'dqn')

    
rewards, belongTo = simulate(expl_param=0.1)  # epsilon 0.1
# checking rewards of ships each step. 

df = pd.DataFrame(rewards).transpose()

df = df.replace(np.nan, '', regex=True)
df.loc[belongTo[0]] # rewards of ships belong to 0: First player
rewards, belongTo = simulate(expl_param=0) 
df = pd.DataFrame(rewards).transpose()

df = df.replace(np.nan, '', regex=True)
df.loc[belongTo[1]] # Second player