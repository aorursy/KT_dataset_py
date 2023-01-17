# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *
from ast import literal_eval

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# I would like to use the hdf format, but I cant see it , therefir here the solution with the csv
#the eproblem is, that the event is just a string, representing the 100*8 array
#it needs therefore to be converted to an array, ugly
dF = pd.read_csv('../input/touch_events.csv')
dF['event'] = dF['event'].apply(literal_eval)

#with the hdf, this sould be find:
#dF = pd.read_hdf('../input/touch_events.hdf')
# Any results you write to the current directory are saved as output.
classes = dF['class'].unique()
fig, axes = subplots(len(classes), figsize=(16,9))

for ax_i, event_t in enumerate(classes):
    axe = fig.axes[ax_i]
    axe.set_title(f'Class : {event_t}')
    dF_class = pd.DataFrame(dF[dF['class']==event_t])
    
    rand_t = np.random.choice(dF_class.index)
    event_dF = pd.DataFrame(dF_class.loc[rand_t]['event'])
    for k_i, k in enumerate(event_dF.keys()):
        axe.plot(event_dF[k]+k_i*2)
fig.savefig('event_overview.jpg')
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

alpha=1e-5 # learning rate, this is a quite normal value
hidden_layer_sizes = (10,10,10) # 3 hidden layers with 10 perceptrons. Systematic test of finding the best paramters sould be done
tol = 1e-12 # this is quite high, but with the small train batch, this sould not be a problem, it will just learn until it got it really right



dF['X'] = dF['event'].apply(lambda e:np.array(e).flatten()) #shrin the 100*8 values to one long vector of the size 800
dF['y'] = dF['class'] # also copy class to y, so we use y as a class description, this seems to be a standard

#now we split all 118 events in two groups. One for training the other one for testing
#we split ~2/3 for train and 1/3 for testing, but try to change this a bit
dF_train, dF_test = train_test_split(dF, test_size=0.3, random_state=42)  





#let there be light
clf = MLPClassifier(solver='lbfgs', alpha=alpha,
                    hidden_layer_sizes=hidden_layer_sizes, random_state=1,tol=tol)

#and turn it on
X = np.vstack(dF_train['X'].values)
y = dF_train['y']
clf.fit(X,y)  

# how does it perform with untrained events?
X_test = np.vstack(dF_test['X'].values)
y_test = dF_test['y']
pred_y = clf.predict(X_test)
dF_test['pred'] = pred_y
score = clf.score(X_test,y_test)
print(score)
print(f"Dude, you get {round(score*100,2)}% right! Let's go bowling.")
print("'Naaa, can't be so good!'")

for r_index, row in dF_test.iterrows():
    print(f'Real event: {row["y"]} - {row["pred"]} : is what the network thinks.')
 
false_pred_dF = dF_test[~(dF_test['y']==dF_test['pred'])]
false_pred_dF

fig, axes = subplots(len(false_pred_dF), figsize=(16,9))

for ax_i,(row_i, row) in enumerate(false_pred_dF.iterrows()):
    axe = fig.axes[ax_i]
    axe.set_title(f'Real: {row["y"]} Pred:{row["pred"]}')
    event_dF = pd.DataFrame(row['event'])
    for k_i, k in enumerate(event_dF.keys()):
        axe.plot(event_dF[k]+k_i*2)
