import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
from sklearn import tree
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler 
np.random.seed(0)
data = pd.read_csv("../input/creditcard.csv")
# This is a function for calculating the F1 Score. 
# A much better way at guaging how well the algorithm 
# is doing than simply by accuracy. 
# Learn more: https://en.wikipedia.org/wiki/F1_score
# Credit to: https://stackoverflow.com/a/45305384
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# This is a little macro to make a confusion matrix as output, nothing special.
from sklearn.metrics import confusion_matrix

def evaluate(model, X_val, Y_val, silent=False):
    predictions = model.predict(X_val)
    predictions = np.around(predictions).flatten()
    results = np.equal(Y_val, predictions)
    acc = float(np.sum(results)/len(results))
    suspected_fraud = np.nonzero(predictions)[0]
    real_fraud = np.nonzero(Y_val)[0]
    fraud_results = results[real_fraud]
    fraud_acc = float(np.sum(fraud_results)/len(fraud_results))
    correct_count = np.sum(fraud_results)
    cm = confusion_matrix(Y_val, predictions)
    if not silent:
        print("       \tPREDICTED")
        print("TRUE  |\tokay\tfraud")
        print("okay  |\t"+str(cm[0][0])+"\t"+str(cm[0][1]))
        print("fraud |\t"+str(cm[1][0])+"\t"+str(cm[1][1]))
        print("overall accuracy: "+str(acc))
        print("fraud accuracy:   "+str(fraud_acc))
    return fraud_acc
#shuffling the data
data = data.sample(frac=1).reset_index(drop=True)
frauds = data[data['Class'] == 1]
validation_portion = 0.15
validation_cutoff_index = int(len(data)*validation_portion)
validation_set = data[:validation_cutoff_index]
training_set = data[validation_cutoff_index:]

training_set = training_set.sample(frac=1).reset_index(drop=True)
validation_set = validation_set.sample(frac=1).reset_index(drop=True)

X=training_set.drop(columns=['Class'])
Y=training_set['Class']
X_val=validation_set.drop(columns=['Class'])
Y_val=validation_set['Class']

print("Training size:   "+str(len(X)))
print(" fraud count: "+str(len(Y[Y==1])))
print("Validation size: "+str(len(X_val)))
print(" fraud count: "+str(len(Y_val[Y_val==1])))

labels = 'Okay', 'Fraud'
sizes = [len(Y[Y==0]), len(Y[Y==1])]
colors = ['green', 'red']
explode = (0.1,0)
# Plot
plt.pie(sizes, labels=labels,
        colors=colors,
       explode=explode)
 
plt.axis('equal')
plt.show()
nn_model = Sequential()
nn_model.add(Dense(100, input_dim=30, activation='relu')) # taking in the 30 inputs
nn_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn_model.add(Dense(300, activation='relu')) # layer of 300 neurons
nn_model.add(Dense(500, activation='sigmoid')) # layer of 500 neurons w/ sigmoid activation
nn_model.add(Dense(1, activation='sigmoid')) # final layer to say if its 

num_epochs = 3
batch_size = 1024
nn_model.compile(loss='logcosh',
                 optimizer=keras.optimizers.RMSprop(lr=0.0001), 
                 metrics=[f1, 'accuracy'])
nn_model.fit(X, Y, epochs=num_epochs, batch_size=batch_size)
evaluate(nn_model, X_val, Y_val)
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X, Y)
evaluate(tree_model, X_val, Y_val)
rf_model = RandomForestClassifier(max_depth=15,
                                  warm_start=False,
                                  n_jobs=-1,
                                  random_state=0)
rf_model.fit(X,Y)
evaluate(rf_model, X_val, Y_val)
okay_val=0.5
fraud_min=0.05
fraud_max=0.951
fraud_incr=0.05
fraud_range=np.arange(fraud_min, fraud_max, fraud_incr)
results = []
for fraud_weight in tqdm(fraud_range):
    rf_model = RandomForestClassifier(max_depth=22,
    #                                 0=okay, 1=fraud
                                      class_weight={0:okay_val, 1:fraud_weight},
                                      warm_start=False,
                                      n_jobs=-1,
                                      random_state=0)
    rf_model.fit(X,Y)
    results.append(evaluate(rf_model, X_val, Y_val, silent=True))
fig, ax = plt.subplots()
ax.plot(fraud_range, results)

ax.set(xlabel='fraud weight', ylabel='fraud_accuracy', title="okay weight = "+str(okay_val))
ax.grid()
plt.show()
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_sample(X, Y)

print("new size:        "+str(len(X_resampled)))
print("new fraud count: "+str(len(X_resampled[Y_resampled==1])))

labels = 'Okay', 'Fraud'
sizes = [len(Y_resampled[Y_resampled==0]), len(Y_resampled[Y_resampled==1])]
colors = ['green', 'red']
explode = (0.1,0)
# Plot
plt.pie(sizes, labels=labels,
        colors=colors,
       explode=explode)
 
plt.axis('equal')
plt.show()
nn2_model = Sequential()
nn2_model.add(Dense(100, input_dim=30, activation='relu')) # taking in the 30 inputs
nn2_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn2_model.add(Dense(200, activation='relu')) # layer of 200 neurons
nn2_model.add(Dense(300, activation='relu')) # layer of 300 neurons
nn2_model.add(Dense(500, activation='sigmoid')) # layer of 500 neurons w/ sigmoid activation
nn2_model.add(Dense(1, activation='sigmoid')) # final layer to say if its 

num_epochs = 3
batch_size = 1024
nn2_model.compile(loss='logcosh',
                 optimizer=keras.optimizers.RMSprop(lr=0.0001), 
                 metrics=[f1, 'accuracy'])
nn2_model.fit(X_resampled, Y_resampled, epochs=num_epochs, batch_size=batch_size)
evaluate(nn2_model, X_val, Y_val)
rf2_model = RandomForestClassifier(max_depth=15,
                                  warm_start=False,
                                  n_jobs=-1,
                                  random_state=0)
rf2_model.fit(X_resampled, Y_resampled)
evaluate(rf2_model, X_val, Y_val)