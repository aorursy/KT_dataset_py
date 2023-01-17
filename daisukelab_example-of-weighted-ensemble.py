import os
import shutil
import pandas as pd
import numpy as np
import keras
from matplotlib import pyplot as plt
%matplotlib inline

ENSEM_TRY = 'E9' # Attempt Id

ensemble_list = ['D6', 'E', 'E4', 'H2']

datadir = "../input/acoustic-scene-2018/" # Original data
exampledir = "../input/tutasc-ensemble-example/" # My examples

# Make label lists
y_labels_train = pd.read_csv(datadir + 'y_train.csv', sep=',')['scene_label'].tolist()
labels = sorted(list(set(y_labels_train)))
label2int = {l:i for i, l in enumerate(labels)}
int2label = {i:l for i, l in enumerate(labels)}

# Load y_valid
y_train_org = [label2int[l] for l in y_labels_train]
splitlist = pd.read_csv(datadir + 'crossvalidation_train.csv', sep=',')['set'].tolist()
y_valid_ref = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'test'])
y_valid_ref = keras.utils.to_categorical(y_valid_ref)
# Optimize weight balance
from scipy import optimize

raw_valid_preds = [np.load(exampledir + 'preds%s4valid.npy' % e) for e in ensemble_list]
ref_valid_cls = [np.argmax(y) for y in y_valid_ref]

def f(weights):
    valid_preds = np.average(raw_valid_preds, axis=0, weights=weights)
    y_valid_pred_cls = [np.argmax(pred) for pred in valid_preds]
    return y_valid_pred_cls

def loss_function(weights):
    y_valid_pred_cls = f(weights)
    n_lost = [result != ref for result, ref in zip(y_valid_pred_cls, ref_valid_cls)]
    #print('loss', np.sum(n_lost) / len(y_valid_pred_cls), 'current weights', weights)
    return np.sum(n_lost) / len(y_valid_pred_cls)

opt_weights = optimize.minimize(loss_function,
                                [1/len(ensemble_list)] * len(ensemble_list),
                                constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                                method= 'Nelder-Mead', #'SLSQP',
                                bounds=[(0.0, 1.0)] * len(ensemble_list),
                                options = {'ftol':1e-10},
                            )['x']

print('Optimum weights = ', opt_weights, 'with loss', loss_function(opt_weights))

def acc_function(weights):
    y_valid_pred_cls = f(weights)
    n_eq = [result == ref for result, ref in zip(y_valid_pred_cls, ref_valid_cls)]
    return np.sum(n_eq) / len(y_valid_pred_cls)

print('Ensembled Accuracy =', acc_function(opt_weights))

# double check answers
n = 5
y_valid_pred_cls = f(opt_weights)
for result, ref in zip(y_valid_pred_cls[:n], ref_valid_cls[:n]):
    print(result, '\t', ref)
# Ensemble submission results
raw_preds = [np.load(exampledir + 'preds%s.npy' % e) for e in ensemble_list]
y_preds = np.average(raw_preds, axis=0, weights=opt_weights) # mean ensemble

with open('submit%s.csv' % ENSEM_TRY, 'w') as f:
    f.writelines(['Id,Scene_label\n'])
    f.writelines(['%d,%s\n' % (i, int2label[np.argmax(pred)]) for i, pred in enumerate(y_preds)])
print('Shape of one model prediction results for validation set', raw_valid_preds[0].shape)
print('Shape of one model prediction results for test set', raw_preds[0].shape)
