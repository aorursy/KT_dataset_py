!pip install kaggle-environments
from kaggle_environments import evaluate, make, utils

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import numpy as np

import pandas as pd

import torch

import matplotlib.pyplot as plt

import time

import datetime

import math

import copy
class ConnectX(nn.Module):

        def __init__(self):

            super().__init__()

            self.fc = nn.Sequential(

                        nn.Linear(42,256, bias=True),

                        nn.Linear(256,7, bias=True),

                        nn.Softmax(dim=1)

                        )



        def forward(self, inputs):

            x = self.fc(inputs)

            return x

        

def init_weights(m):

    if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):

        torch.nn.init.xavier_uniform(m.weight)

        m.bias.data.fill_(0.00)            

            

def return_random_agents(num_agents):

    

    agents = []

    for _ in range(num_agents):

        

        agent = ConnectX()

        

        for param in agent.parameters():

            param.requires_grad = False

            

        init_weights(agent)

        agents.append(agent)

    

    return agents



def run_agents(agents,show_game=True):

    

    reward_agents = []

    

    env = make("connectx", debug=True)



    trainer = env.train([None, "random"])



    for agent in agents:

        agent.eval()



        observation = trainer.reset()



        r=0

        

        tini=datetime.datetime.now()



        while True:



            if(show_game):

                trainer.render()



            inp = torch.tensor(observation.board).type('torch.FloatTensor').view(1,-1)

            output_probabilities = agent(inp).detach().numpy()[0]



            # k = np.where(np.array(observation.board) == 0)[0]

            candidates = np.array([0,1,2,3,4,5,6])



            action = np.random.choice(candidates, 1, p=output_probabilities).item()

            try: 

                observation, reward, done, info = trainer.step(action)



                if(done):

                    if reward == 1: # Won

                        reward = 20

                    elif reward == 0: # Lost

                        reward = -20

                    else: # Draw

                        reward = 10    

                    break

                else:

                    reward = 0.5

                    r=r+reward

            except:

                pass        



        reward_agents.append(r)    

        

    return reward_agents



def return_average_score(agent, runs,show_game=True):

    score = 0.

    for i in range(runs):

        score_temp=run_agents([agent],show_game)[0]

        score += score_temp

        #print("Score for run",i,"has been",score_temp)

    return score/runs



def run_agents_n_times(agents, runs,show_game=True):

    avg_score = []

    for agent in agents:

        avg_score.append(return_average_score(agent,runs,show_game))

    return avg_score



def mutate(agent):



    child_agent = copy.deepcopy(agent)

    

    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf

    

    for param in child_agent.parameters():

        

        if(len(param.shape)==4): #weights of Conv2D



            for i0 in range(param.shape[0]):

                for i1 in range(param.shape[1]):

                    for i2 in range(param.shape[2]):

                        for i3 in range(param.shape[3]):

                            

                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()

                                

                                    



        elif(len(param.shape)==2): #weights of linear layer

            for i0 in range(param.shape[0]):

                for i1 in range(param.shape[1]):

                    

                    param[i0][i1]+= mutation_power * np.random.randn()

                        



        elif(len(param.shape)==1): #biases of linear layer or conv layer

            for i0 in range(param.shape[0]):

                

                param[i0]+=mutation_power * np.random.randn()

        





    return child_agent





def cruce(mother_agent,father_agent):

    

    child_agent = copy.deepcopy(mother_agent)

    

    dim_father=[]

    for j in range(len(list(father_agent.parameters()))):

        dim_father.append(list(father_agent.parameters())[j].shape)





    

    for param in child_agent.parameters():

        for j in range(len(dim_father)):

            if(dim_father[j]==param.shape):

                idx=j

        

        

        if(len(param.shape)==4): #weights of Conv2D



            for i0 in range(param.shape[0]):

                for i1 in range(param.shape[1]):

                    for i2 in range(param.shape[2]):

                        for i3 in range(param.shape[3]):

                            if np.random.uniform(0,1) <= 0.5:

                                param[i0][i1][i2][i3]= list(father_agent.parameters())[idx][i0][i1][i2][i3]

                                

                                    



        elif(len(param.shape)==2): #weights of linear layer

            for i0 in range(param.shape[0]):

                for i1 in range(param.shape[1]):

                    if np.random.uniform(0,1) <= 0.5:

                        param[i0][i1]= list(father_agent.parameters())[idx][i0][i1]



                        



        elif(len(param.shape)==1): #biases of linear layer or conv layer

            for i0 in range(param.shape[0]):

                if np.random.uniform(0,1) <= 0.5:

                    param[i0]= list(father_agent.parameters())[idx][i0]

        





    return child_agent    



def return_children(agents, sorted_parent_indexes, elite_index):

    

    children_agents = []

    

    print(datetime.datetime.now(),"Start: Crossing and Muting agents...")

    

    max_idx=1

    while max_idx<=len(sorted_parent_indexes):

        

        for i in range(max_idx):

            children=cruce(mother_agent=agents[sorted_parent_indexes[i]],father_agent=agents[max_idx])

            children_agents.append(mutate(children))

            if(len(children_agents)>=(num_agents-1)):

                break

        

        max_idx=max_idx+1

           

    

    print(datetime.datetime.now(),"End: Crossing and Muting agents...")

    

    mutated_agents=[]

   

    #now add one elite

    elite_child = add_elite(agents, sorted_parent_indexes, elite_index)

    children_agents.append(elite_child)

    elite_index=len(children_agents) -1 #it is the last one

    

    return children_agents, elite_index







def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):

    

    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    

    if(elite_index is not None):

        candidate_elite_index = np.append(candidate_elite_index,[elite_index])

        

    top_score = None

    top_elite_index = None

    

    print(datetime.datetime.now(),"Start: Playing candidates to elite...")

    for i in candidate_elite_index:



        if(i%10==0):

            show= False # True

        else:

            show=False

        

        score = return_average_score(agents[i],runs=3,show_game=show)

        print("Score for elite i ", i, " is on average", score)

        

        if(top_score is None):

            top_score = score

            top_elite_index = i

        elif(score > top_score):

            top_score = score

            top_elite_index = i

    

    print(datetime.datetime.now(),"End: Playing candidates to elite...")    

    print("Elite selected with index ",top_elite_index, " and average score", top_score)

    

    child_agent = copy.deepcopy(agents[top_elite_index])

    return child_agent







def softmax(x):

    """Compute softmax values for each sets of scores in x."""

    return np.exp(x) / np.sum(np.exp(x), axis=0)

torch.set_grad_enabled(False)
num_agents = 500

agents = return_random_agents(num_agents)

top_limit = int((-1 + np.sqrt(1 + 4*2*num_agents))/2)+1 

generations = 3 #1000

elite_index = None
for generation in range(generations):



    # return rewards of agents

    rewards = run_agents_n_times(agents, 1,show_game=False) #return average of 3 runs

    

    # sort by rewards

    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit]

    

    top_rewards = []

    for best_parent in sorted_parent_indexes:

        top_rewards.append(rewards[best_parent])

    

    print("Generation ", generation, " | Mean rewards all players: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))

    print("Top ",top_limit," scores", sorted_parent_indexes)

    print("Rewards for top: ",top_rewards)

    

    # setup an empty list for containing children agents

    children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)

    print("elite index:",elite_index)

    # kill all agents, and replace them with their children

    agents = children_agents
layers = []



for param in agents[elite_index].parameters():

    if(len(param.shape)==2):

        weights = param.numpy().transpose().tolist()

    if(len(param.shape)==1):

        bias = param.numpy().tolist()



        layers.extend([

            weights,

            bias

        ])     



# Convert all layers into usable form before integrating to final agent

precision = 7



layers = list(map(

    lambda x: str(list(np.round(x, precision))) \

        .replace('array(', '').replace(')', '') \

        .replace(' ', '') \

        .replace('\n', ''),

    layers

))



layers = np.reshape(layers, (-1, 2))



# Create the agent

my_agent = '''def my_agent(observation, configuration):

    import numpy as np



'''



# Write hidden layers

for i, (w, b) in enumerate(layers[:-1]):

    my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)

    my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)

    

# Write output layer

my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(layers[-1][0])

my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(layers[-1][1])



my_agent += '''

    state = observation.board[:]

    out = np.array(state, dtype=np.float32)



'''



# Calculate hidden layers

for i in range(len(layers[:-1])):

    my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)

    

# Calculate output layer

my_agent += '    out = np.matmul(out, ol_w) + ol_b\n'

my_agent += '    out = np.exp(out) / np.sum(np.exp(out), axis=0)\n'



my_agent += '''

    for i in range(configuration.columns):

        if observation.board[i] != 0:

            out[i] = -1e7



    return int(np.argmax(out))

    '''





with open('submission_dn.py', 'w') as f:

    f.write(my_agent)

from submission_dn import my_agent



def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



print("My Agent vs. Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))

print("My Agent vs. Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))

print("Random Agent vs. My Agent:", mean_reward(evaluate("connectx", ["random", my_agent], num_episodes=10)))

print("Negamax Agent vs. My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=10)))