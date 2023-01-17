# Importing some libraries
import pandas as pd
import numpy as np

# Import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Import from sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
url = "/kaggle/input/pima-indians-diabetes-database/diabetes.csv"
pima_data = pd.read_csv(url)
pima_data.shape
#Setting a random seed
seed = 7
np.random.seed(seed)

# Seperating predictors and response variable.
X = pima_data.iloc[:,0:8]
y = pima_data.iloc[:,8]
# Create model 
model = Sequential()
# First hidden layer with 12 neurons.
model.add(Dense(12, input_dim = 8, kernel_initializer = 'uniform', activation = 'relu'))
# Second hidden layer with 8 neurons
model.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
# Output layer with 11 neuron.
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fit the model
model.fit(X, y, validation_split = 0.33, epochs = 150, batch_size = 10)
scores = model.evaluate(X,y)
print("%s %.2f%%" %(model.metrics_names[1], scores[1] * 100))
#Setting a random seed
seed = 7
np.random.seed(seed)

def create_model(kernel_initializer = 'glorot_uniform', optimizer = 'rmsprop'):
    model = Sequential()
    model.add(Dense(12, input_dim = 8, kernel_initializer = kernel_initializer, activation = 'relu'))
    model.add(Dense(8, kernel_initializer = kernel_initializer, activation = 'relu'))
    model.add(Dense(1, kernel_initializer = kernel_initializer, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model

# Create a model from KerasClassifier
model = KerasClassifier(build_fn = create_model, verbose = 0)

# Grid Search epochs, batch_size and optimizer.
optimizers = ['adam','rmsprop']
kernel_initializer = ['glorot_uniform', 'normal','uniform']
epochs = (50,100,150)
batches = (5,10,20)

param_grid = dict(optimizer = optimizers, kernel_initializer = kernel_initializer,
                 nb_epoch = epochs, batch_size = batches)
grid = GridSearchCV(estimator = model, param_grid = param_grid)
grid_result = grid.fit(X,y)

print("Best search %f using %s" %(grid_result.best_score_, grid_result.best_params_))
