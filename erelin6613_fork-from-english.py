from IPython.display import YouTubeVideo

YouTubeVideo('6vYJyOGKCHE', width=800, height=450)
!pip install kaggle-environments --upgrade --quiet
from kaggle_environments import make, evaluate

env = make("halite", debug=True)

env.run(["random", "random", "random", "random"])

env.render(mode="ipython", width=800, height=600)
from kaggle_environments import make

from kaggle_environments.envs.halite.helpers import *



board_size = 5

environment = make("halite", 

                   configuration={

                       "size": board_size, 

                       "startingHalite": 1000})

agent_count = 2

environment.reset(agent_count)

state = environment.state[0]



board = Board(state.observation, 

              environment.configuration)



def move_ships_north_agent(

    observation, configuration):

    board = Board(

        observation, configuration)

    current_player = board.current_player

    for ship in current_player.ships:

        ship.next_action = ShipAction.NORTH

    return current_player.next_actions



environment.reset(agent_count)

environment.run([move_ships_north_agent, "random"])

environment.render(mode="ipython", width=500, height=450)
%%writefile submission.py

import time

import copy

import sys

import math

import collections

import pprint

import numpy as np

import scipy.optimize

import scipy.ndimage

from kaggle_environments.envs.halite.helpers import *

import kaggle_environments

import random



CONFIG_MAX_SHIPS=20

all_actions=[ShipAction.NORTH, ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]

all_dirs=[Point(0,1), Point(1,0), Point(0,-1), Point(-1,0)]

start=None

num_shipyard_targets=4

size=None

ship_target={}

me=None

did_init=False

quiet=False

C=None

class Obj:

  pass

turn=Obj()

turns_optimal=np.array(

  [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],

   [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],

   [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],

   [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],

   [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],

   [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],

   [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],

   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],

   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],

   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])



#### Functions

def print_enemy_ships(board):

  print('\nEnemy Ships')

  for ship in board.ships.values():

    if ship.player_id != me.id:

      print('{:6}  {} halite {}'.format(ship.id,ship.position,ship.halite))

      

def print_actions(board):

  print('\nShip Actions')

  for ship in me.ships:

    print('{:6}  {}  {} halite {}'.format(ship.id,ship.position,ship.next_action,ship.halite))

  print('Shipyard Actions')

  for sy in me.shipyards:

    print('{:6}  {}  {}'.format(sy.id,sy.position,sy.next_action))



def print_none(*args):

  pass



def compute_max_ships(step):

  if step < 200:

    return CONFIG_MAX_SHIPS

  elif step < 300:

    return CONFIG_MAX_SHIPS-2

  elif step < 350:

    return CONFIG_MAX_SHIPS-4

  else:

    return CONFIG_MAX_SHIPS-5



def set_turn_data(board):

  turn.num_ships=len(me.ships)

  turn.max_ships=compute_max_ships(board.step)

  turn.total_halite=me.halite

  turn.halite_matrix=np.reshape(board.observation['halite'], (board.configuration.size,board.configuration.size))

  turn.num_shipyards=len(me.shipyards)

  turn.EP,turn.EH,turn.ES=gen_enemy_halite_matrix(board)

  turn.taken={}

  turn.last_episode = (board.step == (board.configuration.episode_steps-2))

  

def init(obs,config):

  global size

  global print

  if hasattr(config,'myval') and config.myval==9 and not quiet:

    pass

  else:

    print=print_none

    pprint.pprint=print_none

  size = config.size



def limit(x,a,b):

  if x<a:

    return a

  if x>b:

    return b

  return x

  

def num_turns_to_mine(C,H,rt_travel):

  if C==0:

    ch=0

  elif H==0:

    ch=turns_optimal.shape[0]

  else:

    ch=int(math.log(C/H)*2.5+5.5)

    ch=limit(ch,0,turns_optimal.shape[0]-1)

  rt_travel=int(limit(rt_travel,0,turns_optimal.shape[1]-1))

  return turns_optimal[ch,rt_travel]



def halite_per_turn(carrying, halite,travel,min_mine=1):

  turns=num_turns_to_mine(carrying,halite,travel)

  if turns<min_mine:

    turns=min_mine

  mined=carrying+(1-.75**turns)*halite

  return mined/(travel+turns), turns

  

def move(pos, action):

  ret=None

  if action==ShipAction.NORTH:

    ret=pos+Point(0,1)

  if action==ShipAction.SOUTH:

    ret=pos+Point(0,-1)

  if action==ShipAction.EAST:

    ret=pos+Point(1,0)

  if action==ShipAction.WEST:

    ret=pos+Point(-1,0)

  if ret is None:

    ret=pos

  return ret % size



def dirs_to(p1, p2, size=21):

  deltaX, deltaY=p2 - p1

  if abs(deltaX)>size/2:

    #we wrap around

    if deltaX<0:

      deltaX+=size

    elif deltaX>0:

      deltaX-=size

  if abs(deltaY)>size/2:

    #we wrap around

    if deltaY<0:

      deltaY+=size

    elif deltaY>0:

      deltaY-=size

  ret=[]

  if deltaX>0:

    ret.append(ShipAction.EAST)

  if deltaX<0:

    ret.append(ShipAction.WEST)

  if deltaY>0:

    ret.append(ShipAction.NORTH)

  if deltaY<0:

    ret.append(ShipAction.SOUTH)

  if len(ret)==0:

    ret=[None]

  return ret, (deltaX,deltaY)



def shipyard_actions():

  for sy in me.shipyards:

    if turn.num_ships < turn.max_ships:

      if turn.total_halite >= 500 and sy.position not in turn.taken:

        sy.next_action = ShipyardAction.SPAWN

        turn.taken[sy.position]=1

        turn.num_ships+=1

        turn.total_halite-=500



def gen_enemy_halite_matrix(board):

  EP=np.zeros((size,size))

  EH=np.zeros((size,size))

  ES=np.zeros((size,size))

  for id,ship in board.ships.items():

    if ship.player_id != me.id:

      EH[ship.position.y,ship.position.x]=ship.halite

      EP[ship.position.y,ship.position.x]=1

  for id, sy in board.shipyards.items():

    if sy.player_id != me.id:

      ES[sy.position.y,sy.position.x]=1

  return EP,EH,ES



def dist(a,b):

  action,step=dirs_to(a, b, size=21) 

  return abs(step[0]) + abs(step[1])



def nearest_shipyard(pos):

  mn=100

  best_pos=None

  for sy in me.shipyards:

    d=dist(pos, sy.position)

    if d<mn:

      mn=d

      best_pos=sy.position

  return mn,best_pos

  

def assign_targets(board,ships):

  old_target=copy.copy(ship_target)

  ship_target.clear()

  if len(ships)==0:

    return

  halite_min=50

  pts1=[]

  pts2=[]

  for pt,c in board.cells.items():

    assert isinstance(pt,Point)

    if c.halite > halite_min:

      pts1.append(pt)

  for sy in me.shipyards:

    for i in range(num_shipyard_targets):

      pts2.append(sy.position)

  C=np.zeros((len(ships),len(pts1)+len(pts2)))

  for i,ship in enumerate(ships):

    for j,pt in enumerate(pts1+pts2):

      d1=dist(ship.position,pt)

      d2,shipyard_position=nearest_shipyard(pt)

      if shipyard_position is None:

        d2=1

      my_halite=ship.halite

      if j < len(pts1):

        v, mining=halite_per_turn(my_halite,board.cells[pt].halite, d1+d2)

      else:

        if d1>0:

          v=my_halite/d1

        else:

          v=0

      if board.cells[pt].ship and board.cells[pt].ship.player_id != me.id:

        enemy_halite=board.cells[pt].ship.halite

        if enemy_halite <= my_halite:

          v = -1000

        else:

          if d1<5:

            v+= enemy_halite/(d1+1)

      C[i,j]=v

  print('C is {}'.format(C.shape))

  row,col=scipy.optimize.linear_sum_assignment(C, maximize=True)

  pts=pts1+pts2

  for r,c in zip(row,col):

    ship_target[ships[r].id]=pts[c]

  print('\nShip Targets')

  print('Ship      position          target')

  for id,t in ship_target.items():

    st=''

    ta=''

    if board.ships[id].position==t:

      st='MINE'

    elif len(me.shipyards)>0 and t==me.shipyards[0].position:

      st='SHIPYARD'

    if id not in old_target or old_target[id] != ship_target[id]:

      ta=' NEWTARGET'

    print('{0:6}  at ({1[0]:2},{1[1]:2})  assigned ({2[0]:2},{2[1]:2}) h {3:3} {4:10} {5:10}'.format(

      id, board.ships[id].position, t, board.cells[t].halite,st, ta))



  return



def make_avoidance_matrix(myship_halite):

  filter=np.array([[0,1,0],[1,1,1],[0,1,0]])

  bad_ship=np.logical_and(turn.EH <= myship_halite,turn.EP)

  avoid=scipy.ndimage.convolve(bad_ship, filter, mode='wrap',cval=0.0)

  avoid=np.logical_or(avoid,turn.ES)

  return avoid



def make_attack_matrix(myship_halite):

  attack=np.logical_and(turn.EH > myship_halite,turn.EP)

  return attack



def get_max_halite_ship(board, avoid_danger=True):

  mx=-1

  the_ship=None

  for ship in me.ships:

    x=ship.position.x

    y=ship.position.y

    avoid=make_avoidance_matrix(ship.halite)

    if ship.halite>mx and (not avoid_danger or not avoid[y,x]):

      mx=ship.halite

      the_ship=ship

  return the_ship



def remove_dups(p):

  ret=[]

  for x in p:

    if x not in ret:

      ret.append(x)

  return ret



def matrix_lookup(matrix,pos):

  return matrix[pos.y,pos.x]



def ship_converts(board):

  if turn.num_shipyards==0 and not turn.last_episode:

    mx=get_max_halite_ship(board)

    if mx is not None:

      if mx.halite + turn.total_halite > 500:

        mx.next_action=ShipAction.CONVERT

        turn.taken[mx.position]=1

        turn.num_shipyards+=1

        turn.total_halite-=500

  for ship in me.ships:

    if ship.next_action:

      continue

    avoid=make_avoidance_matrix(ship.halite)

    z=[matrix_lookup(avoid,move(ship.position,a)) for a in all_actions]

    if np.all(z) and ship.halite > 500:

      ship.next_action=ShipAction.CONVERT

      turn.taken[ship.position]=1

      turn.num_shipyards+=1

      turn.total_halite-=500

      print('ship id {} no escape converting'.format(ship.id))

    if turn.last_episode and ship.halite > 500:

      ship.next_action=ShipAction.CONVERT

      turn.taken[ship.position]=1

      turn.num_shipyards+=1

      turn.total_halite-=500

      

def ship_moves(board):

  ships=[ship for ship in me.ships if ship.next_action is None]

  assign_targets(board,ships)

  actions={}

  for ship in ships:

    if ship.id in ship_target:

      a,delta = dirs_to(ship.position, ship_target[ship.id],size=size)

      actions[ship.id]=a

    else:

      actions[ship.id]=[random.choice(all_actions)]

      

  for ship in ships:

    action=None

    x=ship.position

    avoid=make_avoidance_matrix(ship.halite)

    attack=make_attack_matrix(ship.halite)

    action_list=actions[ship.id]+[None]+all_actions

    for a in all_actions:

      m=move(x,a)

      if attack[m.y,m.x]:

        print('ship id {} attacking {}'.format(ship.id,a))

        action_list.insert(0,a)

        break

    action_list=remove_dups(action_list)

    for a in action_list:

      m=move(x,a)

      if avoid[m.y,m.x]:

        print('ship id {} avoiding {}'.format(ship.id,a))

      if m not in turn.taken and not avoid[m.y,m.x]:

        action=a

        break

    ship.next_action=action

    turn.taken[m]=1

    

def agent(obs, config):

  global size

  global start

  global prev_board

  global me

  global did_init

  #Do initialization 1 time

  start_step=time.time()

  if start is None:

    start=time.time()

  if not did_init:

    init(obs,config)

    did_init=True

  board = Board(obs, config)

  me=board.current_player

  set_turn_data(board)

  print('==== step {} sim {}'.format(board.step,board.step+1))

  print('ships {} shipyards {}'.format(turn.num_ships,turn.num_shipyards))

  print_enemy_ships(board)

  ship_converts(board)

  ship_moves(board)

  shipyard_actions()

  print_actions(board)

  print('time this turn: {:8.3f} total elapsed {:8.3f}'.format(time.time()-start_step,time.time()-start))

  return me.next_actions

env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])



env.run(["/kaggle/working/submission.py", "random"])

env.render(mode="ipython", width=800, height=600)
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_curve

import matplotlib.pyplot as plt
# assuming we have times damaged, time alive and speed of clicking as Nx3 matrix

# and we are given some simple feedback 0 to degrade difficulty, 1 to upgrade

# we will skip scaling step for now

seed = 13

N = 1000

game_stats = np.random.randint(0, 10, N*3).reshape(N, 3)

target = np.random.choice([0, 1], size=N).reshape(N, 1)

plt.hist(target);
X_train, X_test, y_train, y_test = train_test_split(

    game_stats, target, test_size=0.25, random_state=seed)



model = LogisticRegression()

model = model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy_score(y_test, preds)
plt.plot(roc_curve(y_test, preds)[0], label='false positive',);

plt.plot(roc_curve(y_test, preds)[1], label='false negative');

plt.legend();
import tensorflow as tf

import tensorflow.keras as K
inputs = K.Input(shape=(3,))

x = K.layers.Dense(32, activation=tf.nn.relu)(inputs)

x = K.layers.Dense(64, activation=tf.nn.relu)(x)

outputs = K.layers.Dense(1, activation=tf.nn.softmax)(x)

model = K.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 5

history = model.fit(X_train, y_train, epochs=epochs)
preds = model.predict(X_test)

accuracy_score(y_test, preds)
import pandas as pd

from surprise import SVD, accuracy, Reader, dataset

from surprise.model_selection import cross_validate



ratings_dict = {'userID': np.random.randint(0, 1000, 100),

                'itemID': np.random.randint(0, 100, 100), # say it is going to be some metric, hp, time_alive etc

                'rating': np.random.randint(0, 5, 100)}

df = pd.DataFrame.from_dict(ratings_dict)



reader = Reader(line_format='user item rating', rating_scale=(1, 5))
class UserSet(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in

                            zip(df['userID'], df['itemID'], df['rating'])]

        self.reader=reader



data = UserSet(df, reader)

algo = SVD()

perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

print(perf)