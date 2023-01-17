# %%writefile DataGenerator.py

# learned from: https://github.com/allenday/contextual-bandit/blob/master/contextual_bandit_sim.ipynb

import numpy as np

class DataGenerator():

    """

    Generate bandit data.

    Defaults:

    K = 2 arms

    D = 2 features/arm

    only K arms with fixed features of dimension D too.

    And if you select an arm => it is put into the sample => so why there are many samples when we only have K fixed arms? => what will be put in the sampled results if user select an arm? => isn't it the arm itself and corresponding rewards?

    

    """

    def __init__(self, K = 2, D = 2):

        self.D = D # dimension of the feature vector

        self.K = K # number of bandits

        self.means = np.random.normal(size=self.K)

        self.stds = 1 + 2*np.random.rand(self.K)

        # generate the weight vectors. Initialioze estimate of feature importance for each arm's features

        self.generate_weight_vectors()

    def generate_weight_vectors(self, loc=0.0, scale=1.0):

        self.W = np.random.normal(loc=loc, scale=scale, size=(self.K, self.D))

    

    def generate_samples(self, n = 1000):

        X = np.random.randint(0, 5, size=(n, self.D))



        # The rewards are functions of the inner products of the feature vectors with current weight estimates

        IP = np.dot(X, self.W.T)

        

        # now get the rewards

        R = np.abs(np.random.normal(self.means + IP, self.stds))

        

        return X, R

    

    # Thompson Smapling

    # basic idea: samples from distribution and compares those values for the arms instead

    def thompson_sampling(self, observed_data):

        return np.argmax(np.random.beta(observed_data[:, 0], observed_data[:, 1]))
# %%writefile OnlineVariance.py

#learned from: https://github.com/allenday/contextual-bandit/

#http://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream

import numpy as np

class OnlineVariance(object):

    """

    Welford's algorithm computes the sample variance incrementally

    ddof: delta degree of freedom (used in the divisor: N - ddof) e.g., for the whole population ddof = 0; for sample of elements ddof = 1;

    """

    def __init__(self, iterable=None, ddof = 1):

        self.ddof, self.n, self.mean, self.M2, self.variance = ddof, 0, 0.0, 0.0, 0.0

        if iterable is not None:

            for datum in iterable:

                self.include(datum)

    def include(self, datum):

        self.n += 1

        self.delta = datum - self.mean

        self.mean += self.delta/self.n

        self.M2 += self.delta*(datum - self.mean)

        self.variance = self.M2/(self.n - self.ddof)

    @property

    def std(self):

        return np.sqrt(self.variance)
# %%writefile PositiveStrategy.py

# learned from: https://github.com/allenday/contextual-bandit/

import numpy as np

# from OnlineVariance import OnlineVariance

class PositiveStrategy(object):

    """

    Positive strategy selector.

    Defaults:

    K = 2 arms

    D = 2 features/arm

    epsilon = 0.05 (learning rate/exploration in this case)

    """

    def __init__(self, K = 2, D = 2, epsilon = 0.05):

        self.K = K

        self.D = D

        self.epsilon = epsilon

        

        self.stats = np.empty((K, D), dtype = object)

        

        for k in range(0, K):

            for d in range(0, D):

                self.stats[k, d] = OnlineVariance(ddof = 0)

                

    def mu(self):

        result = np.zeros((self.K, self.D))

        for k in range(0, self.K):

            for d in range(0, self.D):

                result[k, d] = self.stats[k, d].mean

        return result

    

    def sigma(self):

        result = np.zeros((self.K, self.D))

        for k in range(0, self.K):

            for d in range(0, self.D):

                result[k, d] = self.stats[k, d].std

        return result

    

    def include(self, arm, features, value):

        for d in range(0, self.D):

            if features[d] > 0:

                self.stats[arm, d].include(value)

                

    def estimate(self, arm, features):

        # why estimate what is the purpose, this return sum([1, 2, 3]*[1, 3, 2]) = sum([1, 6, 6]) = 13 => so the estimate is the dot product of the features and the arm feature.

        # So there is a context feature and an arm feature

        # OK, sample feature is there, how about arm feature? Where do you get it? And if it is selected => the sample should include the arm feature too?

        # how about time t => at that round => what do you present to the agent?

        return np.sum(features * [val for val in map(lambda x: np.random.normal(x.mean, x.std if x.std > 0 else 1), self.stats[arm])])

    

    def rmse(self, weights):

        # it is the root means squared error of two matrices of size (K, D), between the estimated W and the actual weights.

        return np.sqrt(np.mean((weights-self.mu())**2)/self.K)
import numpy as np

class Simulator(object):

    """

    Simulate model

    epsilon=0.05 learning rate, this is the exploration rate.

    """

    def __init__(self, model, epsilon=0.05):

        self.model = model

        self.K = model.K

        self.D = model.D

        self.epsilon = epsilon



    def simulate(self,features,rewards,weights):

        N = int(rewards.size/self.K)



        regret = np.zeros((N,1))

        rmse = np.zeros((N,1))



        for i in range(0,N):

            F = features[i]

            R = rewards[i]

            

            #known reward and correct choice

            armOptimal = np.argmax(R)

            #estimate the values of the arms and select the armChoice

            armChoice = np.argmax([self.model.estimate(k, F) for k in range(self.K)])

            

            #learn from an arm other than best estimate with p=epsilon

            learn = np.random.uniform() <= self.epsilon

            if learn:

                armAlt = armChoice

                while (armAlt == armChoice):

                    armAlt = int(np.random.uniform() * self.K)

                armChoice = armAlt



            #calculate reward and regret for chosen arm

            armReward = R[armChoice]

            armMaxReward = R[armOptimal]

            armRegret = armMaxReward - armReward

            regret[i] = armRegret

            rmse[i]   = self.model.rmse(weights)



            #reward/penalize accordingly

            if armRegret == 0:

                self.model.include(armChoice, F, armReward)

            else:

                self.model.include(armChoice, F, -1 * armRegret)

            

        return regret, rmse
%matplotlib inline

import numpy as np

from matplotlib import pylab as plt
num_arms = 3

num_features = 5



# define number of samples and number of choices

num_samples = 1000

num_batches = 100

num_experiments = 10

dg = DataGenerator(num_arms, num_features)
total_regret = []

total_rmse = []

for e in range(0, num_experiments):

    print("experiment: %d" % e)

    positiveStrategy = PositiveStrategy(num_arms, num_features)

    simulator = Simulator(positiveStrategy)

    

    previous_rmse = 0.

    for b in range(0, num_batches):

        (sample_features, sample_rewards) = dg.generate_samples(num_samples)

        regret, rmse = simulator.simulate(sample_features, sample_rewards, dg.W)

        

        if previous_rmse == 0:

            initial_rmse = rmse[0][-1]

            previous_rmse = rmse[0][-1]

        if(len(total_rmse) == 0):

            total_rmse = rmse

            total_regret = regret

        else:

            total_rmse += rmse

            total_regret += regret

        

mean_regret = total_regret/num_experiments

mean_rmse = total_rmse/num_experiments
print(dg.W)
print(positiveStrategy.mu())
print(len(sample_rewards))
plt.semilogy(np.cumsum(mean_regret)/num_experiments)

plt.title('Simulated Bandit Performance for K = ' + str(num_arms))

plt.ylabel('Cumulative Expected Regret')

plt.xlabel('Round Index')
plt.semilogx(mean_rmse/num_experiments)

plt.title('Simulated Bandit Performance for K = ' +str(num_arms))

plt.ylabel('RMSE')

plt.xlabel('Round Index')

plt.show()