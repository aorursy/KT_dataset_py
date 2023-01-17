# Import libraries
from keras.models import Sequential  # Keras deep learning library
from keras.layers import Dense 
from keras.optimizers import Adam, SGD
%matplotlib inline 
import matplotlib.pyplot as plt # Matplotlib library: plot graphics
import numpy as np # Numpy library: numerical linear algebra.
import pandas as pd # Pandas library: data processing, CSV file I/O (e.g. pd.read_csv)
import os # Operating System library: file management functionalities
print(os.listdir("../input")) # In Kaggle, a database associated to a kernel is located in "../input"
df = pd.read_csv('../input/weight-height.csv') # Read csv file with weight and height data
df[['Height']] = df[['Height']].values/39.37 # Convert inches to meters
df[['Weight']] = df[['Weight']].values/2.205 # Convert pounds to kilos
display(df.head()) # Shows the first lines of the table containing weights and heights
X = df[['Height']].values # Input feature: person height. X is a numpy array
display(X) # Shows X
y_true = df[['Weight']].values # Actual output (ground truth): person weight. y_true is a numpy array
# A linear regressor can be implemented with one neural network with one dense layer. See https://keras.io/layers/core/#dense
linear_regressor = Sequential() # Create a simple feed-forward neural network
linear_regressor.add(Dense(1, input_shape=(1,))) # Add one dense layer. The model will take as input arrays of shape (batch_size, input_dim)=(*, 1)
linear_regressor.summary() # Shows
# Selection of cost funtion (mean square error) and optimization algorithm (Adam, an alternative to gradient descent)
linear_regressor.compile(loss='mse', optimizer=Adam(lr=0.8))
# Training. 
iterations = 40 # Try with 10, 20, 30, 40, 50,..., 100
linear_regressor.fit(X,y_true, epochs=iterations, batch_size=100)

y_pred = linear_regressor.predict(X) # Prediction using trained model
df.plot(kind = 'scatter', # Plot weight and height values in the dataset.
       x = 'Height',
       y = 'Weight', title = 'Weight and Height in adults')
plt.plot(X, y_pred, color = 'red', linewidth = 3) # Plot predicted heithgt values for input weight values. 
w,b = linear_regressor.get_weights() # Get model parameters
display([w,b])
height = [1.72]
weight = linear_regressor.predict(height) # Prediction using trained model
display(weight)