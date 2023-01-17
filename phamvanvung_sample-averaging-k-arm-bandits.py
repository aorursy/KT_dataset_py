import numpy as np

def bandit(A, q):

    R = np.random.normal(q[A], 1)

    

    return R

def selectBandit(epsilon, excludeCurrentMax=False):

    if(np.random.random()>epsilon):

        return np.argmax(Q)

    else:

        return np.random.randint(0, len(Q))    
num_exps = 2000

num_timesteps = 1000

epsilon = 0.0

rhist = np.zeros(num_timesteps)

ahist = np.zeros(num_timesteps)



for k in range(num_exps):

    Q = np.zeros(10)

    N = np.zeros(10)

    q = np.array([np.random.standard_normal() for i in range(10)])

    optimal_action=np.argmax(q)

    reward_history = []

    optimal_action_history = []

    for i in range(num_timesteps):

        # choose an action

        A = selectBandit(epsilon)

        R = bandit(A, q)

        reward_history.append(R)

        if A == optimal_action:

            optimal_action_history.append(1)

        else:

            optimal_action_history.append(0)

        N[A] += 1

        Q[A] += 1/N[A]*(R - Q[A])

    

    rhist += np.array(reward_history)

    ahist += np.array(optimal_action_history)
rhist /= np.float(num_exps)

ahist /= np.float(num_exps)
import matplotlib.pyplot as plt

plt.plot(rhist, label='average reward, eps = ' + str(epsilon))

plt.legend()

plt.show()
plt.plot(ahist, label='average optimal action, eps = ' + str(epsilon))

plt.legend()

plt.show()