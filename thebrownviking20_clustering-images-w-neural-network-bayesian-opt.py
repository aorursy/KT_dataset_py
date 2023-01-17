# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
import concurrent.futures
import time
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
# Loading training and test set
train = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')
train.head()
# Combining training and test set to get over 70k samples
new_train = train.drop(columns=['label'])
new_test = test.drop(columns=['label'])
som_data = pd.concat([new_train, new_test], ignore_index=True).values
labels = pd.concat([train['label'], test['label']], ignore_index=True).values
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(som_data[i].reshape(28, 28))
plt.show()
#Initializing the map
start_time = time.time()
# The map will have x*y = 50*50 = 2500 features  
som = MiniSom(x=50,y=50,input_len=som_data.shape[1],sigma=0.5,learning_rate=0.4)
# There are two ways to train this data
# train_batch: Data is trained in batches
# train_random: Random samples of data are trained. Following line of code provides random weights as we are going to use train_random for training
som.random_weights_init(som_data)
# Training data for 1000 iterations
som.train_random(data=som_data,num_iteration=1000)
# Finally plotting the map
with concurrent.futures.ProcessPoolExecutor() as executor:
    rcParams['figure.figsize'] = 25, 20
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ['o','s','p','*','^','1','h','x','+','d']
    colors = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#670BE8','#F29F60','#8E1C4A','#85809B']
    for i,x in enumerate(som_data):
        w = som.winner(x)
        plot(w[0]+0.5,w[1]+0.5,markers[labels[i]],markeredgecolor=colors[labels[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
    savefig("map.png")
    show()
end_time = time.time() - start_time
print(int(end_time),"seconds taken to complete the task.")
start_time = time.time()
# Returns a matrix where the element i,j is the number of time that the neuron i,j have been winner.
act_res = som.activation_response(som_data)
# Returns a dictionary wm where wm[(i,j)] is a list with all the patterns that have been mapped in the position i,j.
winner_map = som.win_map(som_data)
# Returns a dictionary wm where wm[(i,j)] is a dictionary that contains the number of samples from a given label that have been mapped in position i,j.
labelmap = som.labels_map(som_data,labels)
end_time = time.time() - start_time
print(int(end_time),"seconds taken to extract data from results.")
sns.heatmap(act_res)
# Extracting outliers
q75, q25 = np.percentile(act_res.flatten(), [75 ,25])
iqr = q75 - q25
lower_fence = q25 - (1.5*iqr)
upper_fence = q75 + (1.5*iqr)
condition = (act_res < lower_fence) | (act_res > upper_fence)
outlier_neurons = np.extract(condition,act_res)
# Plotting the distribution of neurons and outliers
f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,5))
ax1.set(xlabel='Distribution of all neurons')
ax2.set(xlabel='Distribution of outliers')
sns.distplot(act_res.flatten(),ax=ax1)
sns.distplot(outlier_neurons,ax=ax2)
plt.close(2)
plt.close(3)
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(winner_map[list(winner_map)[1]][i].reshape(28, 28))
plt.show()
# Reorganizing the data
train = train[(train.label == 4) | (train.label == 8)]
test = test[(test.label == 4) | (test.label == 8)]

opt_train = train.drop(['label'],axis=1)
opt_test = test.drop(['label'],axis=1)
opt_data = pd.concat([opt_train, opt_test], ignore_index=True).values
labels = pd.concat([train['label'], test['label']], ignore_index=True).values
# Setting some parameters in advance
x = y = 10
input_len = opt_data.shape[1]
sigma = 1.0
learning_rate = 0.5
iterations = 1000
# Now, a function to plot maps
def make_map(x,y,input_len,sigma,learning_rate,iterations):
    som = MiniSom(x=x,y=y,input_len=input_len,sigma=sigma,learning_rate=learning_rate)
    som.random_weights_init(opt_data)
    som.train_random(data=opt_data,num_iteration=iterations)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        rcParams['figure.figsize'] = 10, 8
        bone()
        pcolor(som.distance_map().T)
        colorbar()
        markers = ['o','s','p','*','^','1','h','x','+','d']
        colors = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#8C9376','#F29F60','#8E1C4A','#85809B']
        for i,x in enumerate(opt_data):
            w = som.winner(x)
            plot(w[0]+0.5,w[1]+0.5,markers[labels[i]],markeredgecolor=colors[labels[i]],markerfacecolor='None',markersize=12,markeredgewidth=2)
        show()
# A simple unoptimized map
make_map(x,y,input_len,sigma,learning_rate,iterations)
# This will optimize sigma to minimize the quantization error
best_params = fmin(
    fn = lambda sig: MiniSom(x=x,y=y,input_len=input_len,sigma=sigma,learning_rate=learning_rate).quantization_error(opt_data),
    space = hp.uniform("sig",0.0009,x/2.0001),
    algo = tpe.suggest,
    verbose=1,
    max_evals = 50)
print("The best sigma value after 50 iterations is {}".format(best_params['sig']))
# Let's see the new optimized map
make_map(x,y,input_len,best_params['sig'],learning_rate,iterations)
space = {
    'sig': hp.uniform('sig',0.001,5.0),
    'learning_rate': hp.uniform('learning_rate',0.001,0.5)
}

def opt_map(space):
    sig = space['sig']
    learning_rate = space['learning_rate']
    val = MiniSom(x=x,y=y,input_len=input_len,sigma=sigma,learning_rate=learning_rate).quantization_error(opt_data)
    #print("Now,the quantization error is {}\n".format(val))
    return {'loss':val, 'status':STATUS_OK}

trials = Trials()
best_params = fmin(fn=opt_map,space=space,algo=tpe.suggest,max_evals=50,trials=trials)
print("The best sigma value after 50 iterations is {}".format(best_params['sig']))
print("The best learning_rate after 50 iterations is {}".format(best_params['learning_rate']))
# The more optimized map 
make_map(x,y,input_len,best_params['sig'],best_params['learning_rate'],iterations)