def play():

    state='D'

    seq='D'

    while state!='A' and state!='G':

        action=random.choice((-1,1))

        state=chr(ord(state)+action)

        seq+=state

    return (seq)
def temporal_difference(walk,probs,alpha=0.005):

    gradient_sum=0

    for i in range(1,len(walk)):

        current_state=walk[i]

        prev_state=walk[i-1]

        gradient=X[prev_state]

        gradient_sum+=gradient

        probs[1:6]=probs[1:6]+alpha*(probs[ord(current_state)-65]-probs[ord(prev_state)-65])*gradient_sum

    return (probs)
import random

import numpy as np



probs=np.array([0,0.5,0.5,0.5,0.5,0.5,1]) #We take an inital weight of 0 for state A, 0.5 for states B,C,D,E,F, 1 for state G.



X={'B':np.array([1,0,0,0,0]),'C':np.array([0,1,0,0,0]),'D':np.array([0,0,1,0,0]),

   'E':np.array([0,0,0,1,0]),'F':np.array([0,0,0,0,1])}



sequences=[]



for i in range(100):

    sequences.append(play())    

    

for walk in sequences:

    probs=temporal_difference(walk,probs)

    

probs=[round(i,2) for i in probs]

print ('Probability of winning from each state is ', probs)    