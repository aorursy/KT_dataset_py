import numpy as np



# a simple unrealistic slot-machine simulator.  All it does is generate a random number, 

# Gaussian distrubution, with specified mean and standard deviation

class Bandit:

    def __init__(self, mean=0.0, std=1.0):

        self._mean = mean

        self._std = std

        

    def pull(self):

        return np.random.randn()*self._std + self._mean

    

# Multi-armed bandit model.

# gauss_params is a sequence of tuples, mean and st. d.

# for example, 

# model = Model((1,.1), (1.01,.1), (1.05,.5), (1.1, .1), (2,.1))

# for a 5-armed bandit with the specified mean/std pairs.



# The idea is to run lots of trials and maximize your payoff, given that as a player, you don't know which of the 

# 5 or so slot machines pays off the best and must try them all.

class Model:

    def __init__(self, *gauss_params):

        self._means = []

        self._Ns = []

        self._bandits = []

        for x in gauss_params:

            try:

                mean = x[0]

                std = x[1]

            except IndexError:

                mean = x[0]

                std = 1

            except TypeError:

                mean = x

                std = 1

            self._bandits.append(Bandit(mean, std))

            self._means.append(0)

            self._Ns.append(0)

        

    def trial(self, N=100, epsilon=0.2, show=True):

        payout = 0

        first_time = True

        for i in range(N):

            if np.random.rand() < epsilon or first_time:

                index = np.random.randint(0, len(self._bandits))

                first_time = False

            else:

                index = np.argmax(self._means)

            bandit = self._bandits[index]

            result = bandit.pull()

            payout += result

            self._Ns[index] += 1

            self._means[index] = (1-1/self._Ns[index]) * self._means[index] + 1/self._Ns[index] * result 

        if(show):

            print("Total payout:", payout)

        return self._means

    

    

    




def make_model(num_bandits=3, min_mean=0, max_mean=1, std=.1):

    params = []

    for i in range(num_bandits):

        params.append((np.random.uniform(min_mean, max_mean), std))

    return Model(*params)    



def grid_search(num_bandits=3, min_mean=0, max_mean=1, std=.1, 

                Ns=[1,2,5,10,20,50,100,200,500,1000], epsilons=[.001,.01,.1,.15,.2,.25, .3, .35, .4, .45, .5], 

                required_accuracy=.05, num_trials=100):

    grid = dict()

    for N in Ns:

        for eps in epsilons:

            for trial in range(num_trials):

                model = make_model(num_bandits=num_bandits, min_mean=min_mean, max_mean=max_mean, std=std) 

                result = model.trial(N, eps, show=False)

                dist = 0

                for i in range(num_bandits):

                    dist = max(dist, np.abs(result[i] - model._bandits[i]._mean))

                key = (N,eps,(dist <= required_accuracy))

                if key in grid:

                    grid[key] += 1

                else:

                    grid[key] = 1

                    

    best_key = None

    best_value = -num_trials

    for N in reversed(Ns):

        for eps in reversed(epsilons):

            key1 = (N, eps, True)

            key2 = (N, eps, False)

            if key2 not in grid:

                best_value = grid[key1]

                best_key = (N, eps)

            elif(key1 in grid and grid[key1]-grid[key2] > best_value):

                best_value = grid[key1] - grid[key2]

                best_key = (N, eps)

    return best_key, best_value

    

    
grid_search(num_bandits=5)
model = Model((1,.1), (1.01,.1), (1.05,.5), (1.1, .1), (2,.1))
model.trial(500, 0.4)