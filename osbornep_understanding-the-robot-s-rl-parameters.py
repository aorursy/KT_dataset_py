import time

import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt



from IPython.display import clear_output
optimal_policy = pd.read_csv('../input/OptimalPolicy_angletol45.csv')

optimal_policy.head()
# Create Quiver plot showing current optimal policy in one cell

optimal_action_list = optimal_policy.copy()



x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values

plt.figure(figsize=(10, 10))

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action', alpha = 0.3)

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

plt.title("Optimal Policy for Given Probabilities")

plt.show()
# Probability Function

def probability(bin_x, bin_y, state_x, state_y, throw_deg):





    #First throw exception rule if person is directly on top of bin:

    if((state_x==bin_x) & (state_y==bin_y)):

        probability = 1

    else:

        

        

        # To accomodate for going over the 0 degree line

        if((throw_deg>270) & (state_x<=bin_x) & (state_y<=bin_y)):

            throw_deg = throw_deg - 360

        elif((throw_deg<90) & (state_x>bin_x) & (state_y<bin_y)):

            throw_deg = 360 + throw_deg

        else:

            throw_deg = throw_deg

            

        # Calculate Euclidean distance

        distance = ((bin_x - state_x)**2 + (bin_y - state_y)**2)**0.5



        # max distance for bin will always be on of the 4 corner points:

        corner_x = [-10,-10,10,10]

        corner_y = [-10,10,-10,10]

        dist_table = pd.DataFrame()

        for corner in range(0,4):

            dist = pd.DataFrame({'distance':((bin_x - corner_x[corner])**2 + (bin_y - corner_y[corner])**2)**0.5}, index = [corner])

            dist_table = dist_table.append(dist)

        dist_table = dist_table.reset_index()

        dist_table = dist_table.sort_values('distance', ascending = False)

        max_dist = dist_table['distance'][0]

        

        distance_score = 1 - (distance/max_dist)





        # First if person is directly horizontal or vertical of bin:

        if((state_x==bin_x) & (state_y>bin_y)):

            direction = 180

        elif((state_x==bin_x) & (state_y<bin_y)):

             direction = 0

        

        elif((state_x>bin_x) & (state_y==bin_y)):

             direction = 270

        elif((state_x<bin_x) & (state_y==bin_y)):

             direction = 90

              

        # If person is north-east of bin:

        elif((state_x>bin_x) & (state_y>bin_y)):

            opp = abs(bin_x - state_x)

            adj = abs(bin_y - state_y)

            direction = 180 +  np.degrees(np.arctan(opp/adj))



        # If person is south-east of bin:

        elif((state_x>bin_x) & (state_y<bin_y)):

            opp = abs(bin_y - state_y)

            adj = abs(bin_x - state_x)

            direction = 270 +  np.degrees(np.arctan(opp/adj))



        # If person is south-west of bin:

        elif((state_x<bin_x) & (state_y<bin_y)):

            opp = abs(bin_x - state_x)

            adj = abs(bin_y - state_y)

            direction =  np.degrees(np.arctan(opp/adj))



        # If person is north-west of bin:

        elif((state_x<bin_x) & (state_y>bin_y)):

            opp = abs(bin_y - state_y)

            adj = abs(bin_x - state_x)

            direction = 90 +  np.degrees(np.arctan(opp/adj))



        direction_score = (45-abs(direction - throw_deg))/45

      

        probability = distance_score*direction_score

        if(probability>0):

            probability = probability

        else:

            probability = 0

        

    return(probability)

    
#Define Q(s,a) table by all possible states and THROW actions initialised to 0

Q_table = pd.DataFrame()

for z in range(0,360):

    throw_direction = int(z)

    for i in range(0,21):

        state_x = int(-10 + i)

        for j in range(0,21):

            state_y = int(-10 + j)

            reward = 0

            Q = pd.DataFrame({'throw_dir':throw_direction,'move_dir':"none",'state_x':state_x,'state_y':state_y,'Q':0, 'reward': reward}, index = [0])

            Q_table = Q_table.append(Q)

Q_table = Q_table.reset_index(drop=True)

print("Q table 1 initialised")



#Define Q(s,a) table by all possible states and MOVE actions initialised to 0



for x in range(0,21):

    state_x = int(-10 + x)

    for y in range(0,21):

        state_y = int(-10 + y)

        for m in range(0,8):

            move_dir = int(m)

            

            # skip impossible moves starting with 4 corners then edges

            if((state_x==10)&(state_y==10)&(move_dir==0)):

                continue

            elif((state_x==10)&(state_y==10)&(move_dir==2)):

                continue

                

            elif((state_x==10)&(state_y==-10)&(move_dir==2)):

                continue

            elif((state_x==10)&(state_y==-10)&(move_dir==4)):

                continue

                

            elif((state_x==-10)&(state_y==-10)&(move_dir==4)):

                continue

            elif((state_x==-10)&(state_y==-10)&(move_dir==6)):

                continue

                

            elif((state_x==-10)&(state_y==10)&(move_dir==6)):

                continue

            elif((state_x==-10)&(state_y==10)&(move_dir==0)):

                continue

                

            elif((state_x==10) & (move_dir == 1)):

                continue

            elif((state_x==10) & (move_dir == 2)):

                continue

            elif((state_x==10) & (move_dir == 3)):

                continue

                 

            elif((state_x==-10) & (move_dir == 5)):

                continue

            elif((state_x==-10) & (move_dir == 6)):

                continue

            elif((state_x==-10) & (move_dir == 7)):

                continue

                 

            elif((state_y==10) & (move_dir == 1)):

                continue

            elif((state_y==10) & (move_dir == 0)):

                continue

            elif((state_y==10) & (move_dir == 7)):

                continue

                 

            elif((state_y==-10) & (move_dir == 3)):

                continue

            elif((state_y==-10) & (move_dir == 4)):

                continue

            elif((state_y==-10) & (move_dir == 5)):

                continue

                 

            else:

                reward = 0

                Q = pd.DataFrame({'throw_dir':"none",'move_dir':move_dir,'state_x':state_x,'state_y':state_y,'Q':0, 'reward': reward}, index = [0])

                Q_table = Q_table.append(Q)

Q_table = Q_table.reset_index(drop=True)

print("Q table 2 initialised")

Q_table.tail()
# Initialise V values for all state-action pairs

Q_table['V'] = 0
# Calculate Probability of each State-Action pair, 1 for movement else use probability function

bin_x = 0

bin_y = 0



prob_list = pd.DataFrame()

for n,action in enumerate(Q_table['throw_dir']):

    # Guarantee 100% probability if movement

    if(action == "none"):

        prob = 1

    # Calculate if thrown

    else:

        prob = probability(bin_x, bin_y, Q_table['state_x'][n], Q_table['state_y'][n], action)

    prob_list = prob_list.append(pd.DataFrame({'prob':prob}, index = [n] ))

prob_list = prob_list.reset_index(drop=True)

Q_table['prob'] = prob_list['prob']
Q_table.head()
# Define start position

start_x = -5

start_y = -5
# Subset the Q table for just this start state and randomly select an action

Q_table[ (Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['move_dir']!="none") ].sample()
a_1 = Q_table[ (Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['move_dir']!="none") ].sample()





move_direction = a_1['move_dir'].iloc[0]

#Map this to actual direction and find V(s) for next state

if(move_direction == 0):

    move_x = 0

    move_y = 1

elif(move_direction == 1):

    move_x = 1

    move_y = 1

elif(move_direction == 2):

    move_x = 1

    move_y = 0

elif(move_direction == 3):

    move_x = 1

    move_y = -1

elif(move_direction == 4):

    move_x = 0

    move_y = -1

elif(move_direction == 5):

    move_x = -1

    move_y = -1

elif(move_direction == 6):

    move_x = -1

    move_y = 0

elif(move_direction == 7):

    move_x = -1

    move_y = 1



new_x = a_1['state_x'].iloc[0]+move_x

new_y = a_1['state_y'].iloc[0]+move_y

    

a_2 = Q_table[ (Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['move_dir']!="none") ].sample()

a_2
# Define start position

start_x = -5

start_y = -5

action_cap = 100



action_table = pd.DataFrame()

for a in range(0,action_cap):

    

    # Introduce 50/50 chance for move or throw action

    rng = np.random.rand()

    if rng<=0.5:

        action_class = "throw"

    else:

        action_class = "move"

    

    # THROW ACTION

    if action_class == "throw":

        # If first action, use start state

        if a==0:

            action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']!="none")].sample()

        # Else new x and y are from previous itneration's output

        else:

            new_x = action['state_x'].iloc[0]

            new_y = action['state_y'].iloc[0]

            action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']!="none")].sample()

    

    # ELSE MOVE ACTION

    else:

        # If first action, use start state

        if a==0:

            action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']=="none")].sample()

        # Else new x and y are from previous itneration's output

        else:

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']=="none")].sample()

            

    action_table = action_table.append(action)

   

    # Break loop if action is a throw

    if action['throw_dir'].iloc[0]!="none":

        break

    else:

        continue

action_table = action_table.reset_index(drop=True)     

action_table.head()

    

    
# Define start position

start_x = -5

start_y = -5

action_cap = 100



epsilon = 0.1



action_table = pd.DataFrame()

for a in range(0,action_cap):

    

    

    rng_epsilon = np.random.rand()



    # If our rng is less than or equal to the epsilon parameter, we randomly select

    if rng_epsilon<=epsilon:

        # Introduce 50/50 chance for move or throw action

        rng = np.random.rand()

        if rng<=0.5:

            action_class = "throw"

        else:

            action_class = "move"



        # THROW ACTION

        if action_class == "throw":

            # If first action, use start state

            if a==0:

                action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']!="none")].sample()

            # Else new x and y are from previous itneration's output

            else:

                new_x = action['state_x'].iloc[0]

                new_y = action['state_y'].iloc[0]

                action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']!="none")].sample()



        # ELSE MOVE ACTION

        else:

            # If first action, use start state

            if a==0:

                action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']=="none")].sample()

            # Else new x and y are from previous itneration's output

            else:

                move_direction = action['move_dir'].iloc[0]

                #Map this to actual direction and find V(s) for next state

                if(move_direction == 0):

                    move_x = 0

                    move_y = 1

                elif(move_direction == 1):

                    move_x = 1

                    move_y = 1

                elif(move_direction == 2):

                    move_x = 1

                    move_y = 0

                elif(move_direction == 3):

                    move_x = 1

                    move_y = -1

                elif(move_direction == 4):

                    move_x = 0

                    move_y = -1

                elif(move_direction == 5):

                    move_x = -1

                    move_y = -1

                elif(move_direction == 6):

                    move_x = -1

                    move_y = 0

                elif(move_direction == 7):

                    move_x = -1

                    move_y = 1



                new_x = action['state_x'].iloc[0]+move_x

                new_y = action['state_y'].iloc[0]+move_y

                action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']=="none")].sample()



    #  If our rng is more than the epsilon parameter, we select the best action ("greedily")

    else:

        # Sort by V, use previous action if not first in episode

        if a==0:

            sorted_actions = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y)].sort_values('V', ascending = False)

        else:

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            sorted_actions = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y)].sort_values('V', ascending = False)

            

        best_action = sorted_actions[sorted_actions['V'] == sorted_actions['V'].iloc[0]]



        # If we only have one best action, simply pick this

        if len(best_action)==1:

            action = sorted_actions.iloc[0]

            

        # Otherwise, if we have multiple "best" actions, we randomly select from these "best" actions

        else:

            rng = np.random.rand()

            if rng<=0.5:

                action_class = "throw"

            else:

                action_class = "move"

            # THROW ACTION

            if action_class == "throw":

                action = best_action[(best_action['throw_dir']!="none")].sample()

            # ELSE MOVE ACTION

            else:

                action = best_action[(best_action['throw_dir']=="none")].sample()

    

    action_table = action_table.append(action)

   

    # Break loop if action is a throw

    if action['throw_dir'].iloc[0]!="none":

        break

    else:

        continue

action_table = action_table.reset_index(drop=True)     

action_table.head()

    

    
def eps_greedy_V(Q_table, epsilon, start_x, start_y, action_num, action):

    a = action_num

    rng_epsilon = np.random.rand()



    # If our rng is less than or equal to the epsilon parameter, we randomly select

    if rng_epsilon<=epsilon:

        # Introduce 50/50 chance for move or throw action

        rng = np.random.rand()

        if rng<=0.5:

            action_class = "throw"

        else:

            action_class = "move"



        # THROW ACTION

        if action_class == "throw":

            # If first action, use start state

            if a==0:

                action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']!="none")].sample()

            # Else new x and y are from previous itneration's output

            else:

                new_x = action['state_x'].iloc[0]

                new_y = action['state_y'].iloc[0]

                action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']!="none")].sample()



        # ELSE MOVE ACTION

        else:

            # If first action, use start state

            if a==0:

                action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']=="none")].sample()

            # Else new x and y are from previous itneration's output

            else:

                move_direction = action['move_dir'].iloc[0]

                #Map this to actual direction and find V(s) for next state

                if(move_direction == 0):

                    move_x = 0

                    move_y = 1

                elif(move_direction == 1):

                    move_x = 1

                    move_y = 1

                elif(move_direction == 2):

                    move_x = 1

                    move_y = 0

                elif(move_direction == 3):

                    move_x = 1

                    move_y = -1

                elif(move_direction == 4):

                    move_x = 0

                    move_y = -1

                elif(move_direction == 5):

                    move_x = -1

                    move_y = -1

                elif(move_direction == 6):

                    move_x = -1

                    move_y = 0

                elif(move_direction == 7):

                    move_x = -1

                    move_y = 1



                new_x = action['state_x'].iloc[0]+move_x

                new_y = action['state_y'].iloc[0]+move_y

                action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']=="none")].sample()



    #  If our rng is more than the epsilon parameter, we select the best action ("greedily")

    else:

        # Sort by V, use previous action if not first in episode

        if a==0:

            sorted_actions = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y)].sort_values('V', ascending = False)

        else:

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            sorted_actions = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y)].sort_values('V', ascending = False)

            

        best_action = sorted_actions[sorted_actions['V'] == sorted_actions['V'].iloc[0]]



        # If we only have one best action, simply pick this

        if len(best_action)==1:

            action = best_action

            

        # Otherwise, if we have multiple "best" actions, we randomly select from these "best" actions

        else:

            rng = np.random.rand()

            if rng<=0.5:

                action_class = "throw"

            else:

                action_class = "move"

            # THROW ACTION

            if action_class == "throw":

                #Add excemption if no throw directions in "best" actions

                if len(best_action[(best_action['throw_dir']!="none")])>0:

                    action = best_action[(best_action['throw_dir']!="none")].sample()

                else:

                    action = best_action[(best_action['throw_dir']=="none")].sample()

            # ELSE MOVE ACTION

            else:

                action = best_action[(best_action['throw_dir']=="none")].sample()



    return(action)



def eps_greedy_Q(Q_table, epsilon, start_x, start_y, action_num, action):

    a = action_num

    rng_epsilon = np.random.rand()



    # If our rng is less than or equal to the epsilon parameter, we randomly select

    if rng_epsilon<=epsilon:

        # Introduce 50/50 chance for move or throw action

        rng = np.random.rand()

        if rng<=0.5:

            action_class = "throw"

        else:

            action_class = "move"



        # THROW ACTION

        if action_class == "throw":

            # If first action, use start state

            if a==0:

                action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']!="none")].sample()

            # Else new x and y are from previous itneration's output

            else:

                new_x = action['state_x'].iloc[0]

                new_y = action['state_y'].iloc[0]

                action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']!="none")].sample()



        # ELSE MOVE ACTION

        else:

            # If first action, use start state

            if a==0:

                action = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y) & (Q_table['throw_dir']=="none")].sample()

            # Else new x and y are from previous itneration's output

            else:

                move_direction = action['move_dir'].iloc[0]

                #Map this to actual direction and find V(s) for next state

                if(move_direction == 0):

                    move_x = 0

                    move_y = 1

                elif(move_direction == 1):

                    move_x = 1

                    move_y = 1

                elif(move_direction == 2):

                    move_x = 1

                    move_y = 0

                elif(move_direction == 3):

                    move_x = 1

                    move_y = -1

                elif(move_direction == 4):

                    move_x = 0

                    move_y = -1

                elif(move_direction == 5):

                    move_x = -1

                    move_y = -1

                elif(move_direction == 6):

                    move_x = -1

                    move_y = 0

                elif(move_direction == 7):

                    move_x = -1

                    move_y = 1



                new_x = action['state_x'].iloc[0]+move_x

                new_y = action['state_y'].iloc[0]+move_y

                action = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y) & (Q_table['throw_dir']=="none")].sample()



    #  If our rng is more than the epsilon parameter, we select the best action ("greedily")

    else:

        # Sort by V, use previous action if not first in episode

        if a==0:

            sorted_actions = Q_table[(Q_table['state_x']==start_x) &  (Q_table['state_y']==start_y)].sort_values('Q', ascending = False)

        else:

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            sorted_actions = Q_table[(Q_table['state_x']==new_x) &  (Q_table['state_y']==new_y)].sort_values('Q', ascending = False)

            

        best_action = sorted_actions[sorted_actions['Q'] == sorted_actions['Q'].iloc[0]]



        # If we only have one best action, simply pick this

        if len(best_action)==1:

            action = pd.DataFrame(best_action)

            

        # Otherwise, if we have multiple "best" actions, we randomly select from these "best" actions

        else:

            rng = np.random.rand()

            if rng<=0.5:

                action_class = "throw"

            else:

                action_class = "move"

            # THROW ACTION

            if action_class == "throw":

                #Add excemption if no throw directions in "best" actions

                if len(best_action[(best_action['throw_dir']!="none")])>0:

                    action = best_action[(best_action['throw_dir']!="none")].sample()

                else:

                    action = best_action[(best_action['throw_dir']=="none")].sample()

            # ELSE MOVE ACTION

            else:

                action = best_action[(best_action['throw_dir']=="none")].sample()

    



    return(action)



# Define start position

start_x = -5

start_y = -5

action_cap = 100



epsilon = 0.1



action_table = pd.DataFrame()

action = None

for a in range(0,action_cap):

    

    action = eps_greedy_V(Q_table, epsilon, start_x, start_y, a, action)

    

    action_table = action_table.append(action)

   

    # Break loop if action is a throw

    if action['throw_dir'].iloc[0]!="none":

        break

    else:

        continue

action_table = action_table.reset_index(drop=True)     

action_table.head()

    

    

    

    
# Define start position

start_x = -5

start_y = -5

action_cap = 100



epsilon = 0.1

alpha = 0.5

gamma = 0.5

V_bin = 0





# Make a copy of the initalised Q table so we don't override this

Q_table_TD = Q_table.copy()



action_table = pd.DataFrame()

action = None

for a in range(0,action_cap):

    

    action = eps_greedy_V(Q_table_TD, epsilon, start_x, start_y, a, action)

    # If action is to throw, use probability to find whether this was successful or not and update accordingly

    if action['throw_dir'].iloc[0]!="none":

        rng_throw = np.random.rand()

        

        if rng_throw <= action['prob'].iloc[0]:

            reward = 1

        else:

            reward = -1

        New_V = action['V'].iloc[0] + alpha*(reward + (gamma* V_bin) - action['V'].iloc[0])

    # If move action, we have guaranteed probability and currently no reward for this

    else:

        New_V = action['V'].iloc[0]

    #Update V value for state based on outcome

    Q_table_TD['V'] = np.where( ((Q_table_TD['state_x'] == action['state_x'].iloc[0]) & 

                                (Q_table_TD['state_y'] == action['state_y'].iloc[0])),New_V, Q_table_TD['V'])

    

    

    action_table = action_table.append(action)

   

    # Break loop if action is a throw

    if action['throw_dir'].iloc[0]!="none":

        break

    else:

        continue

action_table = action_table.reset_index(drop=True)     

action_table.head()

    

    

    

    
Q_table_TD.sort_values('V',ascending=False).drop_duplicates(['state_x', 'state_y', 'Q', 'reward','V']).head()
# Define start position

start_x = -5

start_y = -5

action_cap = 10000



epsilon = 0.1

alpha = 0.5

gamma = 0.5

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_TD = Q_table.copy()



action_table = pd.DataFrame()

best_states_table = pd.DataFrame()

for e in range(0,num_episodes):

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_V(Q_table_TD, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            New_V = action['V'].iloc[0] + alpha*(reward + (gamma* V_bin) - action['V'].iloc[0])

        # If move action, we have guaranteed probability and currently no reward for this

        else:

            New_V = action['V'].iloc[0]

        #Update V value for state based on outcome

        Q_table_TD['V'] = np.where( ((Q_table_TD['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_TD['state_y'] == action['state_y'].iloc[0])),New_V, Q_table_TD['V'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_states = Q_table_TD[Q_table_TD['V']!=0].sort_values('V', ascending=False).drop_duplicates(['state_x', 'state_y', 'Q', 'reward','V'])

    best_states['episode'] = e

    

    best_states_table = best_states_table.append(best_states)

best_states_table = best_states_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_value = best_states_table[(best_states_table['state_x']==start_x) & (best_states_table['state_y']==start_y)][['episode','V']].sort_values('episode')

                                         

state_values = Q_table_TD.drop_duplicates(['state_x','state_y'])

    

    
best_states_table.head(10)
best_states_table.tail(10)
start_state_value.head()
plt.plot(start_state_value['episode'], start_state_value['V'])

plt.title("Value of Start State by Episode")

plt.show()
state_values.head()
state_values[["state_y", "state_x", "V"]].pivot("state_y", "state_x", "V")
sns.set(rc={'figure.figsize':(11.7,8.27)})

pivot = state_values[["state_y", "state_x", "V"]].pivot("state_y", "state_x", "V")



ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())



plt.title("State Values V(s)")

ax.invert_yaxis()
sns.set(rc={'figure.figsize':(15,10)})

pivot = state_values[["state_y", "state_x", "V"]].pivot("state_y", "state_x", "V")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("State Values V(s)")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_value['episode'], start_state_value['V'])

ax2.set_title("Value of Start State by Episode")



plt.show()

# Define start position

start_x = -5

start_y = -5

action_cap = 10000



epsilon = 0.1

alpha = 0.5

gamma = 0.5

V_bin = 0



num_episodes = 1000



# Make a copy of the initalised Q table so we don't override this

Q_table_TD = Q_table.copy()



action_table = pd.DataFrame()

best_states_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_V(Q_table_TD, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            New_V = action['V'].iloc[0] + alpha*(reward + (gamma* V_bin) - action['V'].iloc[0])

        # If move action, we have guaranteed probability and currently no reward for this

        else:

            New_V = action['V'].iloc[0]

        #Update V value for state based on outcome

        Q_table_TD['V'] = np.where( ((Q_table_TD['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_TD['state_y'] == action['state_y'].iloc[0])),New_V, Q_table_TD['V'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_states = Q_table_TD[Q_table_TD['V']!=0].sort_values('V', ascending=False).drop_duplicates(['state_x', 'state_y', 'Q', 'reward','V'])

    best_states['episode'] = e

    

    best_states_table = best_states_table.append(best_states)

best_states_table = best_states_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_value = best_states_table[(best_states_table['state_x']==start_x) & (best_states_table['state_y']==start_y)][['episode','V']].sort_values('episode')

                                         

state_values = Q_table_TD.drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_values[["state_y", "state_x", "V"]].pivot("state_y", "state_x", "V")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("State Values V(s)")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_value['episode'], start_state_value['V'])

ax2.set_title("Value of Start State by Episode")



plt.show()





    
# Define start position

start_x = -5

start_y = -5

action_cap = 10000



epsilon = 0.1

alpha = 0.5

gamma = 0.5

V_bin = 0



num_episodes = 1000



# Make a copy of the initalised Q table so we don't override this

Q_table_TD = Q_table.copy()



action_table = pd.DataFrame()

best_states_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_V(Q_table_TD, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            New_V = action['V'].iloc[0] + alpha*(reward + (gamma* V_bin) - action['V'].iloc[0])

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            New_V = action['V'].iloc[0] + alpha*(reward + (gamma* V_bin) - action['V'].iloc[0])

        #Update V value for state based on outcome

        Q_table_TD['V'] = np.where( ((Q_table_TD['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_TD['state_y'] == action['state_y'].iloc[0])),New_V, Q_table_TD['V'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_states = Q_table_TD[Q_table_TD['V']!=0].sort_values('V', ascending=False).drop_duplicates(['state_x', 'state_y', 'Q', 'reward','V'])

    best_states['episode'] = e

    

    best_states_table = best_states_table.append(best_states)

best_states_table = best_states_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_value = best_states_table[(best_states_table['state_x']==start_x) & (best_states_table['state_y']==start_y)][['episode','V']].sort_values('episode')

                                         

state_values = Q_table_TD.drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_values[["state_y", "state_x", "V"]].pivot("state_y", "state_x", "V")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("State Values V(s)")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_value['episode'], start_state_value['V'])

ax2.set_title("Value of Start State by Episode")



plt.show()





    
# Define start position

start_x = -5

start_y = -5

bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.5

gamma = 0.5

V_bin = 0



num_episodes = 1000



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==start_x) & (best_actions_table['state_y']==start_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    
sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Start State")

ax2.plot(best_actions_table[['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode")



plt.show()

optimal_action_start_state = best_actions_table[(best_actions_table['state_x']==start_x)&(best_actions_table['state_y']==start_y)].sort_values('episode',ascending=False)

if (optimal_action_start_state['throw_dir'].iloc[0]=="none"):

    print("The optimal action from the start state is to MOVE in direction: ", optimal_action_start_state['move_dir'].iloc[0])

else:

    print("The optimal action from the start state is to THROW in direction: ", optimal_action_start_state['throw_dir'].iloc[0])

# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



optimal_action_list = state_action_values



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))

                                             

# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))

                                             

x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values



plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

plt.title("Optimal Policy for Given Probabilities")

plt.show()
#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.5

gamma = 0.5

V_bin = 0



num_episodes = 1000



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    

    # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

    start_x = np.random.randint(-10,11)

    start_y = np.random.randint(-10,11)

    

    

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Base State")

ax2.plot(best_actions_table[['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode")





plt.show()
# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



optimal_action_list = state_action_values



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))

                                             

# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))

                                             

x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values



plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

plt.title("Optimal Policy for Given Probabilities")

plt.show()
#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.9

gamma = 0.5

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    

    # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

    start_x = -5

    start_y = -5

    

    

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Base State")

ax2.plot(best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode, alpha = 0.9")



plt.show()
#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.1

gamma = 0.5

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    

    # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

    start_x = -5

    start_y = -5

    

    

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Base State")

ax2.plot(best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode, alpha = 0.1")



plt.show()
#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

gamma = 0.5

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()





best_actions_table = pd.DataFrame()

for alp in range(1,11):

    alpha = alp/10



    action_table = pd.DataFrame()

    for e in range(0,num_episodes):

        clear_output(wait=True)

        print("Current alpha: ", alpha)

        print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")



        # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

        start_x = -5

        start_y = -5





        action = None

        for a in range(0,action_cap):



            action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

            # If action is to throw, use probability to find whether this was successful or not and update accordingly

            if action['throw_dir'].iloc[0]!="none":

                rng_throw = np.random.rand()



                if rng_throw <= action['prob'].iloc[0]:

                    reward = 1

                else:

                    reward = -1

                bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

                bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

                New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

            # If move action, we have guaranteed probability and no introduce a small positive reward

            else:

                reward = 0.1

                move_direction = action['move_dir'].iloc[0]

                #Map this to actual direction and find V(s) for next state

                if(move_direction == 0):

                    move_x = 0

                    move_y = 1

                elif(move_direction == 1):

                    move_x = 1

                    move_y = 1

                elif(move_direction == 2):

                    move_x = 1

                    move_y = 0

                elif(move_direction == 3):

                    move_x = 1

                    move_y = -1

                elif(move_direction == 4):

                    move_x = 0

                    move_y = -1

                elif(move_direction == 5):

                    move_x = -1

                    move_y = -1

                elif(move_direction == 6):

                    move_x = -1

                    move_y = 0

                elif(move_direction == 7):

                    move_x = -1

                    move_y = 1



                new_x = action['state_x'].iloc[0]+move_x

                new_y = action['state_y'].iloc[0]+move_y

                next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

                next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']





                New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



            #Update Q(s,a) value for state based on outcome

            Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                        (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                        (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                        (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                        ),New_Q, Q_table_Q['Q'])



            #Add column to denote which episode this is for

            action['episode'] = e

            action_table = action_table.append(action)



            # Break loop if action is a throw

            if action['throw_dir'].iloc[0]!="none":

                break

            else:

                continue

        action_table = action_table.reset_index(drop=True)  



        #Find best states

        best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

        best_actions['episode'] = e

        best_actions['alpha'] = alpha



        best_actions_table = best_actions_table.append(best_actions)

    best_actions_table = best_actions_table.reset_index(drop=True)

    #Produce Summary output for each episode so we can observe convergence

    start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')



    state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])



best_actions_table.head(10)
actions_plot_data = best_actions_table[best_actions_table['Q']!=0][['episode','alpha','Q']].groupby(['alpha','episode']).mean().reset_index()

actions_plot_data.head(10)
import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

import plotly.offline as py

py.init_notebook_mode(connected=True)





def InterAnim(Param_col, x_col, y_col, marker_size=None, plot_title=None, title_size=None, x_label=None,

              y_label=None, param_label=None, plot_type=None, marker_col=None, marker_alpha=None, fig_size_auto=None, fig_width=None, fig_height=None, vert_grid=None, horiz_grid=None ):



    # Need format: param(slider)/repeats(x)/output(y)

    ourData = pd.DataFrame()

    if (Param_col is None)|(x_col is None)|(y_col is None):

        print("Please provide data inputs: Parameter column, x column and y column.")

    else:

        ourData['year'] = Param_col

        ourData['lifeExp'] = x_col

        ourData['gdpPercap'] = y_col

    

    ourData['continent'] = ''

    ourData['country'] = ''

    

    

    # SET DEFAULT PARAMETERS

    if (marker_size is None):

        marker_size = 1

    else:

        marker_size = marker_size

    ourData['pop'] = 50000*marker_size



    # Find parameter intervals

    alpha = list(set(ourData['year']))

    alpha = np.round(alpha,1)

    alpha = np.sort(alpha)[::-1]

    years = np.round([(alpha) for alpha in alpha],1)



    

    if (plot_title is None):

        plot_title = ""

    else:

        plot_title = plot_title

    

    if (title_size is None):

        title_size = 24

    else:

        title_size = title_size

        

    if (x_label is None):

        x_label = ""

    else:

        x_label = x_label

        

    if (y_label is None):

        y_label = ""

    else:

        y_label = y_label



    if (param_label is None):

        param_label = ""

    else:

        param_label = param_label



    if (plot_type is None):

        plot_type = "markers"

    else:

        plot_type = plot_type    

        

    if (marker_col is None):

        marker_col = "rgb(66, 134, 244)"

    else:

        marker_col = marker_col    

    

    if (marker_alpha is None):

        marker_alpha = 0.8

    else:

        marker_alpha = marker_alpha 

        

    if (fig_size_auto is None):

        fig_size_auto = True

    else:

        fig_size_auto = fig_size_auto

        

    if (fig_size_auto  is  False) & (fig_width is None):

        fig_width = 1500

    else:

        fig_width = fig_width

        

    if (fig_size_auto  is  False) & (fig_height is None):

        fig_height = 1500

    else:

        fig_height = fig_height

        

    if (vert_grid is None):

        vert_grid = True

    else:

        vert_grid = vert_grid

    

    if (horiz_grid is None):

        horiz_grid = True

    else:

        horiz_grid = horiz_grid

        

        

    ## Apply Method for creating animation

    dataset = ourData

    continents = []

    for continent in dataset['continent']:

        if continent not in continents:

            continents.append(continent)

    # make figure

    figure = {

        'data': [],

        'layout': {},

        'frames': []

    }

    # fill in most of layout

    figure['layout']['title'] = {'text': plot_title, 'font':{'size':title_size}}

    figure['layout']['xaxis'] = {'range': [ min(dataset['lifeExp']) - (min(dataset['lifeExp'])/10),

                                            max(dataset['lifeExp']) + (max(dataset['lifeExp'])/10)  ], 'title': x_label, 'showgrid':vert_grid }

    figure['layout']['yaxis'] = {'range': [ min(dataset['gdpPercap']) - (min(dataset['gdpPercap'])/10),

                                            max(dataset['gdpPercap']) + (max(dataset['gdpPercap'])/10) ],'title': y_label, 'type': 'linear', 'showgrid':horiz_grid}

    figure['layout']['hovermode'] = 'closest'

    

    figure['layout']['autosize'] = fig_size_auto

    figure['layout']['width'] = fig_width

    figure['layout']['height'] = fig_height

    

    figure['layout']['sliders'] = {

        'args': [

            'transition', {

                'duration': 900,

                'easing': 'cubic-in-out'

            }

        ],

        'initialValue': '1952',

        'plotlycommand': 'animate',

        'values': years,

        'visible': True

    }

    figure['layout']['updatemenus'] = [

        {

            'buttons': [

                {

                    'args': [None, {'frame': {'duration': 500, 'redraw': False},

                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                    'label': 'Play',

                    'method': 'animate'

                },

                {

                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                    'transition': {'duration': 0}}],

                    'label': 'Pause',

                    'method': 'animate'

                }

            ],

            'direction': 'left',

            'pad': {'r': 10, 't': 87},

            'showactive': False,

            'type': 'buttons',

            'x': 0.1,

            'xanchor': 'right',

            'y': 0,

            'yanchor': 'top'

        }

    ]





    sliders_dict = {

        'active': 0,

        'yanchor': 'top',

        'xanchor': 'left',

        'currentvalue': {

            'font': {'size': 20},

            'prefix': param_label,

            'visible': True,

            'xanchor': 'right'

        },

        'transition': {'duration': 300, 'easing': 'cubic-in-out'},

        'pad': {'b': 10, 't': 50},

        'len': 0.9,

        'x': 0.1,

        'y': 0,

        'steps': []

    }

    # make data

    year = years[0]

    for continent in continents:

        dataset_by_year = dataset[np.round(dataset['year'],1) == np.round(year,1)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]

        data_dict = {

            'x': list(dataset_by_year_and_cont['lifeExp']),

            'y': list(dataset_by_year_and_cont['gdpPercap']),

            'mode': plot_type,

            'text': list(dataset_by_year_and_cont['country']),

            'marker': {

                'sizemode': 'area',

                'sizeref': 100,

                'size': list(dataset_by_year_and_cont['pop'])

            },

            'name': continent

        }

        figure['data'].append(data_dict)

    # make frames

    for year in years:

        frame = {'data': [], 'name': str(year)}

        for continent in continents:

            dataset_by_year = dataset[np.round(dataset['year'],1) == np.round(year,1)]

            dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]

            data_dict = {

                'x': list(dataset_by_year_and_cont['lifeExp']),

                'y': list(dataset_by_year_and_cont['gdpPercap']),

                'mode': plot_type,

                'text': list(dataset_by_year_and_cont['country']),

                'opacity':marker_alpha,

                'marker': {

                    'sizemode': 'area',

                    'sizeref': 100,

                    'size': list(dataset_by_year_and_cont['pop']),

                    'color':marker_col,

                },

                'name': continent

            }

            frame['data'].append(data_dict)

        figure['frames'].append(frame)

        slider_step = {'args': [

            [year],

            {'frame': {'duration': 700, 'redraw': False},

             'mode': 'immediate',

           'transition': {'duration': 700}}

         ],

         'label': year,

         'method': 'animate'}

        sliders_dict['steps'].append(slider_step)



    figure['layout']['sliders'] = [sliders_dict]

    

    return(figure)

Param_col = actions_plot_data['alpha']

x_col = actions_plot_data['episode']

y_col = actions_plot_data['Q']



marker_size = 0.7



plot_title = "Interactive-Animation Parameter Optimisation of RL Convergence - Varying Alpha"

title_size = 28

x_label = "Episode"

y_label = "Mean Q for all non-zero states"

param_label = "Alpha = "



#plot_type = 'markers', 'lines+markers' or 'lines'

plot_type = 'markers'



# color could also be hex code

marker_col = 'rgb(17, 157, 255)'

marker_alpha = 0.8



fig_size_auto = False

fig_width = 1500

fig_height = 700



# Gridlines

vert_grid = False

horiz_grid = True

animation_figure = InterAnim(

                            # REQUIRED

                            Param_col=Param_col, x_col=x_col, y_col=y_col,



                            # OPTIONAL AESTHETICS

                             marker_size=marker_size, plot_title=plot_title, title_size=title_size,

                             x_label=x_label, y_label=y_label, param_label=param_label, plot_type=plot_type,

                             marker_col=marker_col, marker_alpha=marker_alpha, fig_size_auto=fig_size_auto, 

                             fig_width=fig_width, fig_height=fig_height, vert_grid=vert_grid, horiz_grid=horiz_grid  )

iplot(animation_figure)

#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.1

gamma = 0.9

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    

    # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

    start_x = -5

    start_y = -5

    

    

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Base State")

ax2.plot(best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode, alpha = 0.1, gamma = 0.9")



plt.show()
#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.1

gamma = 0.1

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    

    # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

    start_x = -5

    start_y = -5

    

    

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Base State")

ax2.plot(best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[best_actions_table['Q']!=0][['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode, alpha = 0.1, gamma = 0.1")



plt.show()
#Change variable name to base for output graphs

base_x = -5

base_y = -5



bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.1

V_bin = 0



num_episodes = 100



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()





best_actions_table = pd.DataFrame()

for gam in range(1,11):

    gamma = gam/10



    action_table = pd.DataFrame()

    for e in range(0,num_episodes):

        clear_output(wait=True)

        print("Current alpha: ", alpha)

        print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")



        # Randomly select start position between: -4,-5,-6 (note the upper bound is soft, i.e. <-3 = <=-4)

        start_x = -5

        start_y = -5





        action = None

        for a in range(0,action_cap):



            action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

            # If action is to throw, use probability to find whether this was successful or not and update accordingly

            if action['throw_dir'].iloc[0]!="none":

                rng_throw = np.random.rand()



                if rng_throw <= action['prob'].iloc[0]:

                    reward = 1

                else:

                    reward = -1

                bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

                bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

                New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

            # If move action, we have guaranteed probability and no introduce a small positive reward

            else:

                reward = 0.1

                move_direction = action['move_dir'].iloc[0]

                #Map this to actual direction and find V(s) for next state

                if(move_direction == 0):

                    move_x = 0

                    move_y = 1

                elif(move_direction == 1):

                    move_x = 1

                    move_y = 1

                elif(move_direction == 2):

                    move_x = 1

                    move_y = 0

                elif(move_direction == 3):

                    move_x = 1

                    move_y = -1

                elif(move_direction == 4):

                    move_x = 0

                    move_y = -1

                elif(move_direction == 5):

                    move_x = -1

                    move_y = -1

                elif(move_direction == 6):

                    move_x = -1

                    move_y = 0

                elif(move_direction == 7):

                    move_x = -1

                    move_y = 1



                new_x = action['state_x'].iloc[0]+move_x

                new_y = action['state_y'].iloc[0]+move_y

                next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

                next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']





                New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



            #Update Q(s,a) value for state based on outcome

            Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                        (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                        (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                        (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                        ),New_Q, Q_table_Q['Q'])



            #Add column to denote which episode this is for

            action['episode'] = e

            action_table = action_table.append(action)



            # Break loop if action is a throw

            if action['throw_dir'].iloc[0]!="none":

                break

            else:

                continue

        action_table = action_table.reset_index(drop=True)  



        #Find best states

        best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

        best_actions['episode'] = e

        best_actions['gamma'] = gamma



        best_actions_table = best_actions_table.append(best_actions)

    best_actions_table = best_actions_table.reset_index(drop=True)

    #Produce Summary output for each episode so we can observe convergence

    start_state_action_values = best_actions_table[(best_actions_table['state_x']==base_x) & (best_actions_table['state_y']==base_y)][['episode','Q']].sort_values('episode')



    state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])



actions_plot_data = best_actions_table[best_actions_table['Q']!=0][['episode','gamma','Q']].groupby(['gamma','episode']).mean().reset_index()

actions_plot_data.head(10)
Param_col = actions_plot_data['gamma']

x_col = actions_plot_data['episode']

y_col = actions_plot_data['Q']



marker_size = 0.7



plot_title = "Interactive-Animation Parameter Optimisation of RL Convergence - Varying Gamma"

title_size = 28

x_label = "Episode"

y_label = "Mean Q for all non-zero states"

param_label = "Gamma = "



#plot_type = 'markers', 'lines+markers' or 'lines'

plot_type = 'markers'



# color could also be hex code

marker_col = 'rgb(204, 51, 255)'

marker_alpha = 0.8



fig_size_auto = False

fig_width = 1500

fig_height = 700



# Gridlines

vert_grid = False

horiz_grid = True

animation_figure = InterAnim(

                            # REQUIRED

                            Param_col=Param_col, x_col=x_col, y_col=y_col,



                            # OPTIONAL AESTHETICS

                             marker_size=marker_size, plot_title=plot_title, title_size=title_size,

                             x_label=x_label, y_label=y_label, param_label=param_label, plot_type=plot_type,

                             marker_col=marker_col, marker_alpha=marker_alpha, fig_size_auto=fig_size_auto, 

                             fig_width=fig_width, fig_height=fig_height, vert_grid=vert_grid, horiz_grid=horiz_grid  )

iplot(animation_figure)

# Define start position

start_x = -5

start_y = -5

bin_x = 0

bin_y = 0

action_cap = 10000



epsilon = 0.1

alpha = 0.1

gamma = 0.9

V_bin = 0



num_episodes = 10000



# Make a copy of the initalised Q table so we don't override this

Q_table_Q = Q_table.copy()



action_table = pd.DataFrame()

best_actions_table = pd.DataFrame()

for e in range(0,num_episodes):

    clear_output(wait=True)

    print("Current Episode: ",  np.round(e/num_episodes,4) *100,"%")

    action = None

    for a in range(0,action_cap):



        action = eps_greedy_Q(Q_table_Q, epsilon, start_x, start_y, a, action)

        # If action is to throw, use probability to find whether this was successful or not and update accordingly

        if action['throw_dir'].iloc[0]!="none":

            rng_throw = np.random.rand()



            if rng_throw <= action['prob'].iloc[0]:

                reward = 1

            else:

                reward = -1

            bin_Q = Q_table_Q[(Q_table_Q['state_x']==bin_x) & (Q_table_Q['state_y']==bin_y)]

            bin_max_Q = bin_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*bin_max_Q)) 

        # If move action, we have guaranteed probability and no introduce a small positive reward

        else:

            reward = 0.1

            move_direction = action['move_dir'].iloc[0]

            #Map this to actual direction and find V(s) for next state

            if(move_direction == 0):

                move_x = 0

                move_y = 1

            elif(move_direction == 1):

                move_x = 1

                move_y = 1

            elif(move_direction == 2):

                move_x = 1

                move_y = 0

            elif(move_direction == 3):

                move_x = 1

                move_y = -1

            elif(move_direction == 4):

                move_x = 0

                move_y = -1

            elif(move_direction == 5):

                move_x = -1

                move_y = -1

            elif(move_direction == 6):

                move_x = -1

                move_y = 0

            elif(move_direction == 7):

                move_x = -1

                move_y = 1



            new_x = action['state_x'].iloc[0]+move_x

            new_y = action['state_y'].iloc[0]+move_y

            next_action_Q = Q_table_Q[(Q_table_Q['state_x']==new_x) &  (Q_table_Q['state_y']==new_y)]

            next_action_max_Q = next_action_Q.sort_values('Q', ascending=False).iloc[0]['Q']

            



            New_Q = ((1-alpha)*action['Q'].iloc[0]) + alpha*(reward + (gamma*next_action_max_Q)) 



        #Update Q(s,a) value for state based on outcome

        Q_table_Q['Q'] = np.where( ((Q_table_Q['state_x'] == action['state_x'].iloc[0]) & 

                                    (Q_table_Q['state_y'] == action['state_y'].iloc[0]) &

                                    (Q_table_Q['throw_dir'] == action['throw_dir'].iloc[0]) &

                                    (Q_table_Q['move_dir'] == action['move_dir'].iloc[0])

                                    ),New_Q, Q_table_Q['Q'])

    

        #Add column to denote which episode this is for

        action['episode'] = e

        action_table = action_table.append(action)



        # Break loop if action is a throw

        if action['throw_dir'].iloc[0]!="none":

            break

        else:

            continue

    action_table = action_table.reset_index(drop=True)  

    

    #Find best states

    best_actions = Q_table_Q[Q_table_Q['Q']!=0].sort_values('Q', ascending=False).drop_duplicates(['state_x', 'state_y'])

    best_actions['episode'] = e

    

    best_actions_table = best_actions_table.append(best_actions)

best_actions_table = best_actions_table.reset_index(drop=True)

#Produce Summary output for each episode so we can observe convergence

start_state_action_values = best_actions_table[(best_actions_table['state_x']==start_x) & (best_actions_table['state_y']==start_y)][['episode','Q']].sort_values('episode')

                                         

state_action_values = Q_table_Q.sort_values('Q',ascending=False).drop_duplicates(['state_x','state_y'])

    

    

sns.set(rc={'figure.figsize':(15,10)})

pivot = state_action_values[["state_y", "state_x", "Q"]].pivot("state_y", "state_x", "Q")



ax = plt.subplot(221)

ax = sns.heatmap(pivot)

ax.hlines(range(-10,21), *ax.get_xlim())

ax.vlines(range(-10,21), *ax.get_ylim())

ax.set_title("Best State-Action Values for each State")

ax.invert_yaxis()



ax2 = plt.subplot(222)

ax2.plot(start_state_action_values['episode'], start_state_action_values['Q'], label = "Start State")

ax2.plot(best_actions_table[['episode','Q']].groupby('episode').mean().reset_index()['episode'],

         best_actions_table[['episode','Q']].groupby('episode').mean().reset_index()['Q'],label='All States')

ax2.legend()

ax2.set_title("Value of State's Best Action by Episode")



plt.show()





optimal_action_start_state = best_actions_table[(best_actions_table['state_x']==start_x)&(best_actions_table['state_y']==start_y)].sort_values('episode',ascending=False)

if (optimal_action_start_state['throw_dir'].iloc[0]=="none"):

    print("The optimal action from the start state is to MOVE in direction: ", optimal_action_start_state['move_dir'].iloc[0])

else:

    print("The optimal action from the start state is to THROW in direction: ", optimal_action_start_state['throw_dir'].iloc[0])



optimal_action_start_state.head()
    

# Create Quiver plot showing current optimal policy in one cell

arrow_scale = 0.1



optimal_action_list = state_action_values



optimal_action_list['Action'] = np.where( optimal_action_list['move_dir'] == 'none', 'THROW', 'MOVE'  )





optimal_action_list['move_x'] = np.where(optimal_action_list['move_dir'] == 0, int(0),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(1),

                                         np.where(optimal_action_list['move_dir'] == 3, int(1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(0),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 7, int(-1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['move_y'] = np.where(optimal_action_list['move_dir'] == 0, int(1),

                                         np.where(optimal_action_list['move_dir'] == 1, int(1),

                                         np.where(optimal_action_list['move_dir'] == 2, int(0),

                                         np.where(optimal_action_list['move_dir'] == 3, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 4, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 5, int(-1),

                                         np.where(optimal_action_list['move_dir'] == 6, int(0),

                                         np.where(optimal_action_list['move_dir'] == 7, int(1),

                                         int(-1000)

                                        ))))))))

optimal_action_list['throw_dir_2'] = np.where(optimal_action_list['throw_dir']=="none",int(-1000), optimal_action_list['throw_dir'])



# Define horizontal arrow component as 0.1*move direction or 0.1/-0.1 depending on throw direction

optimal_action_list['u'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_x']*arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==0, 0,np.where(optimal_action_list['throw_dir_2']==180, 0,

                                    np.where(optimal_action_list['throw_dir_2']==90, arrow_scale ,np.where(optimal_action_list['throw_dir_2']==270, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']<180, arrow_scale,-arrow_scale))))))

                                             

# Define vertical arrow component based 0.1*move direciton or +/- u*tan(throw_dir) accordingly

optimal_action_list['v'] = np.where(optimal_action_list['Action']=="MOVE", optimal_action_list['move_y']*arrow_scale, 

                                    np.where(optimal_action_list['throw_dir_2']==0, arrow_scale,np.where(optimal_action_list['throw_dir_2']==180, -arrow_scale,

                                    np.where(optimal_action_list['throw_dir_2']==90, 0,np.where(optimal_action_list['throw_dir_2']==270, 0,

                                    optimal_action_list['u']/np.tan(np.deg2rad(optimal_action_list['throw_dir_2'].astype(np.float64))))))))

                                             

x = optimal_action_list['state_x']

y = optimal_action_list['state_y']

u = optimal_action_list['u'].values

v = optimal_action_list['v'].values



plt.figure(figsize=(10, 10))

plt.quiver(x,y,u,v,scale=0.5,scale_units='inches')

sns.scatterplot( x="state_x", y="state_y", data=optimal_action_list,  hue='Action')

plt.title("Optimal Policy for Given Probabilities")

plt.show()