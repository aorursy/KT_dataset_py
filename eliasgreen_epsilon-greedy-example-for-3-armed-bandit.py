import numpy as np 

import matplotlib.pyplot as plt 
# Action class

class Action: 

    def __init__(self, true_value): 

        self.true_value = true_value 

        self.mean_value = 0

        self.N = 0

  

    # method for getting true value with some "noise"

    def choose(self):  

        return np.random.randn() + self.true_value 

  

    # method for estimating true value

    def update(self, got_value): 

        self.N += 1

        self.mean_value = (1 - 1.0 / self.N)*self.mean_value + (1.0 / self.N)*got_value 
def run_experiment(true_value_1, true_value_2, true_value_3, eps, decrease_rate, N):  

    actions = [Action(true_value_1), Action(true_value_2), Action(true_value_3)] 

    data = np.empty(N) 

    

    for i in range(N): 

        # get random probability

        p = np.random.random() 

        

        # select action

        if p < eps: 

            j = np.random.choice(3) 

        else: 

            j = np.argmax([action.mean_value for action in actions]) 

        x = actions[j].choose() 

        actions[j].update(x) 



        # update epsilon

        eps -= decrease_rate



        # for the plot 

        data[i] = x 

        

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1) 



    # plot results

    plt.plot(cumulative_average) 

    plt.plot(np.ones(N)*true_value_1) 

    plt.plot(np.ones(N)*true_value_2) 

    plt.plot(np.ones(N)*true_value_3) 

    plt.xscale('log') 

    plt.show() 



    for i in range(len(actions)): 

        print('Estimated value for', i+1, 'action:', round(actions[i].mean_value, 2))

  

    return cumulative_average 
experiment = run_experiment(1.0, 1.1, 1.5, 1, 0.01, 100000) 
experiment = run_experiment(100.0, 20.0, 30.0, 1, 0.0001, 100000) 