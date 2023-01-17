import numpy as np
import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, GRU, Flatten, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
%matplotlib inline  
# Display test set precision recall curve
def display_precision_recall_curve(reference, score):
    """
    Function to display the precision recall for a reference set.
    
    Arguments:
    reference -- the reference labels given for the set
    score -- the score computed 

    Returns:
    null
    """
    average_precision = average_precision_score(reference, score)
    precision, recall, _ = precision_recall_curve(reference, score)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
def model(input_shape):
    """
    Function creating the Exo_detector model.
    
    Arguments:
    input_shape -- shape of the input

    Returns:
    model -- a model instance in Keras
    """
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV + MaxPool layer to detect patterns
    X = Conv1D(32, kernel_size=10, strides=4)(X_input)
    X = MaxPooling1D(pool_size=4, strides=2)(X)
    X = Activation('relu')(X)
    
    # Step 2: GRU Layer
    X = GRU(192,return_sequences=True)(X)
    X = Flatten()(X)
    
    # Final sigmoid activation layer
    X = Dropout(0.5)(X)                                 
    X = BatchNormalization()(X)    
    X = Dense(1, activation="sigmoid")(X)

    model = Model(inputs= X_input, outputs = X)
    
    return model  
# get traincsv files
train = pd.read_csv("../input/exoTrain.csv")
nx, m = train.shape
m -= 1

y_train = train['LABEL'].values
y_train -=1

X_train = train.drop('LABEL', axis=1).values.reshape(nx, m, 1)

print("Train set: \nNumber of examples={0}\nNumber of readings={1}".format(nx, m))
# get testcsv files
test = pd.read_csv("../input/exoTest.csv")
ny, my = test.shape
my -= 1

y_test = test['LABEL'].values
y_test -= 1

X_test = test.drop('LABEL',axis=1).values.reshape(ny, m, 1)

assert m == my, "Error: train and test set have not the same number of timesteps"

print("Train set: \nNumber of examples={0}\nNumber of readings={1}".format(ny, my))
# Set a seed for reproducibility
seed(42)
set_random_seed(42)

# prepare model
gru_model = model(input_shape = (m, 1))
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
gru_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

# check model shapes
gru_model.summary()
# raise positive examples weight because of skewed classes and fit the model
class_weight = {0: 1., 1: 10}
gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=4, shuffle=True, class_weight=class_weight)
# Predict scores with trained model
y_train_score = gru_model.predict(X_train).flatten()
# Display precision_recall_curve for train set
display_precision_recall_curve(y_train, y_train_score)
# Predict scores for test set
y_test_score = gru_model.predict(X_test).flatten()
# Display test set precision recall curve
display_precision_recall_curve(y_test, y_test_score)
# Compute f_scores and choose best threshold value
f_scores = []
for i in range(800):
    f_scores.append(f1_score(y_test,np.where(y_test_score > i/1000,1,0)))
imax = np.argmax(f_scores)
y_max = np.where(y_test_score > imax/1000,1,0)

threshold = np.median((np.where(f_scores == f_scores[imax])[0]))/1000
print('Best Threshold {0:0.3f}, fscore {1:0.5f} , precision {2:0.5f} , recall {3:0.5f}'.format(imax/1000, f_scores[imax], precision_score(y_test, y_max), recall_score(y_test, y_max)))

plt.figure(figsize=(16,10))
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.xlim([0.0, threshold*2000])
plt.ylim([0.0, 1.01])
plt.plot(f_scores)
# Look at predicted positive labels:
yh = np.where(y_test_score>=threshold,1,0)
print("F1_Score {0:4f} at threshold {1:4f}\n".format(f1_score(y_test, yh), threshold))

print("Predicted ExoPlanet samples", np.where(y_test_score>=threshold)[0])
print("Actual ExoPlanet Samples", np.where(y_test==1)[0])