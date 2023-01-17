import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
%matplotlib inline  

from tensorflow import set_random_seed
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
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
    Function creating the Fraud_detector model.
    
    Arguments:
    input_shape -- shape of the input

    Returns:
    model -- a model instance in Keras
    """
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV + MaxPool layer to detect patterns
    X = BatchNormalization()(X_input)
    X = Dense(128, activation="tanh")(X)
    X = Dense(128, activation="tanh")(X)
    X = Dense(128, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Dense(1, activation="sigmoid")(X)

    model = Model(inputs= X_input, outputs = X)
    
    return model  
df = pd.read_csv("../input/creditcard.csv")
df.head()
#Create dataframes of only Fraud and Normal transactions.
Fraud = df[df.Class == 1]
Normal = df[df.Class == 0]

# Set X_train equal to 80% of the fraudulent transactions.
X_train = Fraud.sample(frac=0.8)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

y_train = X_train['Class']
X_train.drop('Class', axis=1, inplace=True)

# X_test contains all the transaction not in X_train.
X_test = df.loc[~df.index.isin(X_train.index)]
y_test = X_test['Class']
X_test.drop('Class', axis=1, inplace=True)

nx, m = X_train.shape

print("Train set: \nNumber of examples={0}\nNumber of features={1}".format(nx, m))
# Set a seed for reproducibility
seed(1)
set_random_seed(1)

# prepare model
model = model(input_shape = (m,))
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

# check model shapes
model.summary()
# raise positive examples weight because of skewed classes and fit the model
class_weight = {0: 1., 1: 10}
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=1, shuffle=True, class_weight=class_weight)
# Predict scores with trained model
y_train_score = model.predict(X_train).flatten()
# Display precision_recall_curve for train set
display_precision_recall_curve(y_train, y_train_score)
# Predict scores for test set
y_test_score = model.predict(X_test).flatten()
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
plt.xlim([0.0, 1000])
plt.ylim([0.0, 1.01])
plt.plot(f_scores)
