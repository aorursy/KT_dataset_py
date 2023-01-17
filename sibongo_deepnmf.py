from time import time
import numpy as np
np.random.seed(5)
start = time()
N = 32
A = 0.03
B = 0.9
A2 = 0.1 # parameters for cluster assignment random draws.
B2 = 0.95 
COMPS_N = 20
DATAPOINTS = 1000000

clusters = (np.random.beta(A, B, N) * 10 for _ in range(COMPS_N))

clusters = np.vstack((cluster for cluster in  clusters))

def get_cluster_assignments(n):
    return np.random.beta(A2, B2, (n, COMPS_N))

assignments = get_cluster_assignments(DATAPOINTS)

simulated_data = np.dot(assignments, clusters)
'Took ', time() - start
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.constraints import non_neg
import numpy as np


k = 30
k_hidden = 48
params_ = {'activation':'selu', 'bias_constraint': non_neg(),
				'kernel_constraint': non_neg()}

xshape = simulated_data.shape[1]
train_on = 900000 # train size

def get_models():
    input_layer = Input(shape=(xshape,))
    #noised_input = GaussianNoise(noise_stdev)(input_layer)
    input_layer_norm = BatchNormalization()(input_layer)
    hidden_layer = Dense(k_hidden, **params_)(input_layer_norm)
    hidden_layer = BatchNormalization()(hidden_layer)
    clustering = Dense(k, name='clustering', **params_)(hidden_layer)
    clustering = BatchNormalization()(clustering)
    output = Dense(xshape, name='representation', **params_)(clustering)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    clustering_model = Model(inputs=input_layer, outputs=clustering)
    clustering_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model, clustering_model

model, clustering_model = get_models()
model.fit(x=simulated_data[:train_on], y=simulated_data[:train_on], shuffle=True,
    batch_size=1024, verbose=1, epochs=10) # removed tbCallBack, and EarlyStopping


print('Evaluation: ', model.evaluate(simulated_data[train_on:], simulated_data[train_on:]))
print('Clustering:?', clustering_model.predict(simulated_data[0:2])[0])
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock as l1
weights = model.get_layer(name='representation').get_weights()[0]
print(weights.shape, clusters.shape)
fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
fig.tight_layout()
iter_weights = (weight for weight in weights)
iter_clusters = (cluster for cluster in clusters)
for weight, cluster in zip(iter_weights, iter_clusters):
    color=np.random.rand(3)
    ax1.plot(range(32), weight, color=color)
    ax2.plot(range(32), cluster / 10, color=color)

for weight in iter_weights:
    ax1.plot(range(32), weight, color=np.random.rand(3))
for cluster in iter_clusters:
    ax2.plot(range(32), cluster / 10, color=np.random.rand(3))
ax1.set_title('Decoder Weights')
ax2.set_title('Cluster Assignments used to generate the data.')
plt.show()
distances = np.empty((15, 20))
for i in range(20):
    for j in range(15):
        distances[j, i] = l1(weights[j], clusters[i])

plt.imshow(distances, cmap="hot"); plt.show()
from collections import defaultdict
def jaccard_index(A, B):
    A = set(A)
    B = set(B)
    return len(A.intersection(B)) / len(A.union(B))
VAL_N = 1000
Y_prob = np.empty((VAL_N, COMPS_N))
Y_dict = defaultdict(list)
Y = np.empty((VAL_N,))
Y_prob = assignments[900000:900000+VAL_N]
for i, asnt in enumerate(Y_prob):
    assignment = np.argmax(asnt)
    Y[i] = assignment
    Y_dict[assignment].append(i)

simulated_X = np.dot(Y_prob, clusters)

Y_pred_dict = defaultdict(list)
Y_pred = np.empty((VAL_N,))
Y_pred_prob = clustering_model.predict(simulated_X)
for i, prob in enumerate(Y_pred_prob):
    assignment = np.argmax(prob)
    Y_pred[i] = assignment
    Y_pred_dict[assignment].append(i)
    
indexes = np.empty((len(Y_pred_dict.keys()), len(Y_dict.keys())))
for i in range(len(Y_pred_dict.keys())):
    for j in range(len(Y_dict.keys())):
        indexes[i, j] = jaccard_index(Y_pred_dict[i], Y_dict[j])

for i, row in enumerate(indexes):
    print(i, row.max(), sep=': ')
