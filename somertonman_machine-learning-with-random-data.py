import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# number of observations (i.e. records,rows) of our sample data

observations = 1000



# random vectors for Xs and Zs

xs = np.random.uniform(low=-10,high=10, size=(observations,1))

zs = np.random.uniform(-10,10,size=(observations,1))



# stacking 2 random data columns together

inputs = np.column_stack((xs,zs))



# checking resulting column shape

print ("Inputs matrix: ",inputs.shape)

print ("Inputs matrix sample: ",inputs)



# targets = f(x,z) = 2x + 3z + 5 + noise



noise = np.random.uniform(-1,1,(observations,1))



targets = 2*xs + 3*zs + 5 + noise



print ("Targets matrix: ",targets.shape)

print ("Targets matrix sample: ",targets[:5])
targets = targets.reshape(observations,)

fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111, projection='3d')

ax.plot(xs,zs,targets)

ax.set_xlabel('X-s')

ax.set_ylabel('Y-s')

ax.set_zlabel('Targets')

ax.view_init(azim=70)

plt.show()

targets = targets.reshape(observations,1)
# initial range for weights and biases

init_range = 0.01



weights = np.random.uniform(-init_range,init_range,size=(2,1))

biases = np.random.uniform(-init_range,init_range,size=1)
# etas for performing grad descent

#learning_rates = [0.1,]

fig=plt.figure(figsize=(10,60))





w = np.random.uniform(-init_range,init_range,size=(2,1))

b = np.random.uniform(-init_range,init_range,size=1)



learning_rates = [round(i/20,3) for i in range(1,7)][::-1]



for n,z in enumerate(learning_rates):

    weights = w

    biases = b

    for i in range(100):

        outputs = np.dot(inputs,weights) + biases

        deltas = outputs - targets

        loss = np.sum(deltas ** 2) / 2 / observations

        deltas_scaled = deltas/observations

        weights = weights - z * np.dot(inputs.T,deltas_scaled)

        biases = biases - z * np.sum(deltas_scaled)

    ax=fig.add_subplot(len(learning_rates),1,n+1)

    ax.plot(outputs,targets)

    ax.set_title("Learning rate:"+str(z))



#print (weights,biases)

#print (outputs,targets)



        

#ax = fig.add_subplot()

#plt.plot(outputs,targets)

print (weights,biases)
fig = plt.figure(figsize=(15,15))

plt.plot (outputs,targets)

plt.xlabel('Outputs')

plt.ylabel('Targets')

plt.show()