import numpy as np
#OpenAI gym for environments
import gym
import random
#create the environment
env = gym.make("FrozenLake-v0");
#get the size of the Q-table i.e. number of states and actions on each state from the env
n_actions = env.action_space.n;
n_states = env.observation_space.n;

#create the Q-table with above dimensions and filled with zeros intially as the agent is unaware of the environment
q_table = np.zeros((n_states,n_actions))
#check to see 16 states with 4 actions in each state
print(q_table.shape)
#number of episodes for the agent to learn from
n_episodes = 20000;
#learning rate used in Bellman equation
learning_rate = 0.9;
#maximum steps to end an episodes
max_steps = 99;
#discounting rate used in Bellman equation
gamma = 0.9;

#Exploration parameters
epsilon = 1.0;
#exploration probability at start of episode
max_epsilon = 1.0;
#minimum exploration probability
min_epsilon = 0.01;
#decay rate for exploration probability
decay = 0.006;
#store rewards
rewards = [];

#let the agent play for defined number of episodes
for episode in range(n_episodes):
    #reset the environment for each episode
    state = env.reset();
    #define initial parameters
    step = 0;
    #to keep track whether the agent dies
    done = False;
    #keep track of rewards at each episode
    total_rewards = 0;
    
    #run for each episode
    for step in range(max_steps):
        #generate a random number between 0,1 for exploration-eploitation tradeoff 
        #i.e. random number > epsilon -> eploitation else exploration 
        #as the agent does'nt know much because epsilon is being lowered at each step
        #its only exploration at the start
        e_e_tradeoff = random.uniform(0,1);
        
        #exploitation
        if(e_e_tradeoff > epsilon):
            #take the maximum reward paying action on the current state
            action  = np.argmax(q_table[state,:]);
            
        #exploration
        else:
            #get a random action
            action = env.action_space.sample();
            
        #take the action and observe the new outcome state and reward
        new_state, reward, done, info  = env.step(action);
        
        #update the state-action reward value in the q-table using the Bellman equation
        #Q(s,a) = Q(s,a) + learning_rate*[Reward(s,a) + gamma*max Q(snew,anew) - Q(s,a)]
        q_table[state,action] = q_table[state,action] + learning_rate * (reward + gamma * np.max(q_table[new_state,:]) - q_table[state,action]);
        
        #add to total rewards for this episode
        total_rewards += reward;
        
        #define new state
        state = new_state;
        
        #end the episode if agent dies
        if(done == True):
            break;
            
    #reduce the epsilon after each episode
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode);
    
    #keep track of total rewards for each episode
    rewards.append(total_rewards);
    
print("Score over time: " + str(sum(rewards)/n_episodes));
print("Q-Table: ");
print(q_table);
        
#reset the state
env.reset()

# lets test for 10 episodes
for episode in range(10):
    state = env.reset();
    step = 0;
    done = False;
    print("XXXXXXXXXX");
    print("Episode: ",episode);
    
    for step in range(max_steps):
        
        #take the action with maximum expected future reward form the q-table
        action = np.argmax(q_table[state,:]);
        
        new_state, reward, done, info = env.step(action);
        
        if done:
            #print only the last stage to check if the agent reached the goal(G) or fell into a hole(H)
            env.render();
            
            print("Number of steps taken: ",step);
            break;
            
        state = new_state;

#close the connection to the environment
env.close();
    
