import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/credit-card-applications/Credit_Card_Applications.csv')
dataset.head()
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values  #We won't be using this value in our training as SOM is an Unsupervised Learning algorithm
X.shape
y.shape
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1)) 
X = sc.fit_transform(X)
! pip install minisom
from minisom import MiniSom
som = MiniSom(x=20 ,y=20 ,sigma=1.0 ,learning_rate=0.5 ,input_len=15)
som.random_weights_init(X)
som.train_random(data=X ,num_iteration=100)
from pylab import bone, pcolor, colorbar, plot, show

plt.figure(figsize=(17, 10), dpi= 80, facecolor='w', edgecolor='k') # To make the fig bigger 

pcolor(som.distance_map().T) # This line finds out the mean inter neuron distance and makes a map based on these distances.
                             # It makes clusters based on the colours based on the distances. The darker the colour the closer the neurons is to it's neighbourhood.
                             # The lighter neurons are the outliers and if customers are present in it that means they are fradulent.

colorbar()                   # This is the legend of the map
plt.figure(figsize=(17, 10), dpi= 80, facecolor='w', edgecolor='k') # To make the fig bigger 
pcolor(som.distance_map().T) 

colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):  # Looping through X such as x loops through the rows and i loops through columns of that customer (each attribute of a customer).
    w = som.winner(x)      # Finding out the winner node of each customer
    plot(w[0] + 0.5,       # w[0]- x coordinate , w[1] - y coordinate. We are placing the markers at the center of each node/neuron. That's the whole code inside the plot().
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
mappings = som.win_map(X)
mappings.keys()
frauds = np.concatenate( (mappings[(1,1)],mappings[(1,3)],mappings[(7,15)],mappings[(5,4)],mappings[(6,4)]) , axis = 0 ) 
frauds = sc.inverse_transform(frauds)
CustID = frauds[:,0]
CustID
customers = dataset.iloc[: , 1: ].values 
customers.shape
is_fraud = np.zeros(len(dataset))

for i in range (len(dataset)) :
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
is_fraud.shape
# Just checking the data 

is_fraud[0:50]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)
import tensorflow as tf 
# TensorFlow version check
tf.__version__
# Initializing the ANN
classifier_ann = tf.keras.models.Sequential() #We will add layers afterwards

# Adding the input layer and first hidden layer
"""nodes = number of output nodes (input nodes are taken care automatically) , activation - activation funct used  """
classifier_ann.add(tf.keras.layers.Dense(units = 2 , activation='relu')) 

# Adding the output layer (We want to have probabilities as output)
"""If no of categories is 3 or more then output_dim = 3 (or more) , activation = softmax""" 
classifier_ann.add(tf.keras.layers.Dense(units = 1 , activation='sigmoid')) 

# Compile ANN (Applying SGD) - The backpropagation step
"""For more than 3 classifiers use loss = categorical_crossentropy"""
classifier_ann.compile(optimizer='adam', loss='binary_crossentropy' , metrics= ['accuracy'] )

# Fitting the ANN to the training set
classifier_ann.fit(customers , is_fraud, batch_size= 1 , epochs= 2)
y_pred = classifier_ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[: , 0:1].values , y_pred) , axis = 1)
np.set_printoptions(precision=4 , suppress = True)
y_pred
y_pred = y_pred[y_pred[:, 1].argsort()]
y_pred
fraud_dataset = pd.DataFrame({'CustomerID': y_pred[: , 0] , 'Probability' : y_pred[: , 1]})
fraud_dataset
fraud_dataset.to_csv("../working/fraud_probability.csv", index=False)