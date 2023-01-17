import numpy as np       # matrices

import scipy.optimize    # optimization routines

import matplotlib.pyplot as plt
def R(n1,n2,m,H):

    return (1-.75**m)*H/(n1+n2+m)



H=500

n1=10

n2=10

r=[]

for m in range(14):

   r.append(R(n1,n2,m,H))



plt.title('For total travel = 20 steps')

plt.xlabel('steps mining')

plt.ylabel('halite per step')

plt.plot(r)
opt=[]

fig=plt.figure(1)

for travel in range(30):

    def h(mine):

        return -R(0,travel,mine,500)  # we put - because we want to maximize R

    res=scipy.optimize.minimize_scalar(h, bounds=(1,15),method='Bounded')

    opt.append(res.x)

plt.plot(opt)    

plt.xlabel('total travel steps')

plt.ylabel('mining steps')

plt.title('Optimal steps for mining by total travel')
def num_turns_to_mine(rt_travel):

  #given the number of steps round trip, compute how many turns we should plan on mining

  if rt_travel <= 1:

    return 2

  if rt_travel <= 2:

    return 3

  elif rt_travel <=4:

    return 4

  elif rt_travel <=7:

    return 5

  elif rt_travel <=12:

    return 6

  elif rt_travel <=19:

    return 7

  elif rt_travel <= 28:

    return 8

  else:

    return 9



ints=[]

for travel in range(30):

    ints.append(num_turns_to_mine(travel))

plt.plot(opt,label='exact')

plt.plot(ints, label='int approx')

plt.legend()

def best_cell(data):

  #given a list of (travel, halite) tuples, determine the best one

  halite_per_turn=[]

  for t,h in data:

    halite_per_turn.append(R(0,t,num_turns_to_mine(t),h))

  mx=max(halite_per_turn)

  idx=halite_per_turn.index(mx)

  mine=num_turns_to_mine(data[idx][0])

  print('best cell is {} for {:6.1f} halite per step, mining {} steps'.format(idx,mx,mine))

data=[(5,200), (7,190), (10,300),(12,500)]                    

best_cell(data)