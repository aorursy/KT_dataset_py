import os

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.neighbors import NearestNeighbors
# Read the data

X_full = pd.read_csv('../input/learn-together/train.csv', index_col='Id')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')



#Drop a couple of useless columns

X_full.drop('Soil_Type7', axis=1, inplace=True)

X_test.drop('Soil_Type7',axis=1, inplace=True)

X_full.drop('Soil_Type15', axis=1, inplace=True)

X_test.drop('Soil_Type15',axis=1, inplace=True)



# Separate target from predictors 

y_full = X_full.Cover_Type

X_full.drop(['Cover_Type'], axis=1, inplace=True)
weights = [

3.3860658430898054,

0.4163438499758126,

7.35783588470092,

1.4635508470705287,

2.512455585483701,

0.7879386244955993,

2.3361452772106412,

4.509437549105931,

1.2565844481748276,

0.8105744594321818,

357.62840785739945,

195.87206818235353]

len(weights)
#Here is the code I used to find the weights.  You can ignore this.



if False:

    cols = list(X_full.columns.values)

    

    #These are my starting weights.  It reflects my view that Wilderness and Soil are very important,

    #so I gave them 10000.  Elevation is also important but contains big numbers, so I start it with 4.

    #Play around with other starting points.

    weights = [4,1,1,1,1,1,1,1,1,1,10000,10000]

    

    best_score_ever=0

    best_ever_wts = [i for i in weights]

    lr = 0.5

    

    for step in range(1000):        

        X_full_copy = X_full.copy()



        for i in range(10):

            c = X_full.columns[i]

            X_full_copy[c] *= weights[i]

        for i in range(10,14):

            c = X_full.columns[i]

            X_full_copy[c] *= weights[10]

        for i in range(14,len(X_full.columns)):

            c = X_full.columns[i]

            X_full_copy[c] *= weights[11]



        r = lr*random.random()

        

        

        # Choose a random weight to change

        train_index = random.randint(12)

        train_col = X_full.columns[train_index]

        

        # We will test four factors for changing the current weight.

        factors = [1-r,1,1+r, 1+2*r]        

        wts = [weights[train_index] * f for f in factors]



        best_score=0

        best_wt=-1



        for wt in wts:  

            if train_index<10:            

                X_full_copy[train_col] = wt * X_full[train_col]

            if train_index==10: 

                for i in range(10,14):

                    c = X_full.columns[i]

                    X_full_copy[c] = wt*X_full[c]

            if train_index > 10: 

                for i in range(14,len(X_full.columns)):

                    c = X_full.columns[i]

                    X_full_copy[c] = wt*X_full[c]



            model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')

            distances, indices  = model.fit(X_full_copy).kneighbors(X_full_copy)

            

            # we use the second best because the first best is itself.

            second_best = indices[:,1]

            labels = y_full.tolist()

            my_labels = [labels[i] for i in second_best]

            score = accuracy_score(labels, my_labels)        



            if score > best_score_ever :

                best_score_ever = score

                best_ever_wts = [i for i in weights]

                best_ever_wts[train_index] = wt

                print("\t\t\t\t\t\t\t\t\t\t\t\t\tnew best ever:",best_score_ever)

            if score > best_score:

                best_wt=wt

                best_score=score



        old_wt =  weights[train_index] 

        

        # Notice that I only go half-way to the new weights.  Just seemed like a good idea, but not sure.

        weights[train_index] =  (weights[train_index]+best_wt)/2

        

        print("step",step,"col",train_index,"best",round(old_wt,2),"->",round(best_wt,2),"\t\t\t\t\t\t\t\t\t\t\t",best_score) 

        # lr*=.999

    print("best weights",best_ever_wts)

    print("final weights",weights)
# Define and fit model, using the sklearn library.

model = KNeighborsClassifier(n_neighbors=1, p=1)



X_full_copy = X_full.copy()

X_test_copy = X_test.copy()



for i in range(10):

    c = X_full.columns[i]

    X_full_copy[c] = weights[i]*X_full_copy[c]

    X_test_copy[c] = weights[i]*X_test_copy[c]

for i in range(10,14):

    c = X_full.columns[i]

    X_full_copy[c] = weights[10]*X_full_copy[c]

    X_test_copy[c] = weights[10]*X_test_copy[c]

for i in range(14,len(X_full.columns)):

    c = X_full.columns[i]

    X_full_copy[c] = weights[11]*X_full_copy[c]

    X_test_copy[c] = weights[11]*X_test_copy[c]



#model.fit(X_train, y_train)

model.fit(X_full_copy, y_full)

preds_full = model.predict(X_full_copy)



#this should give 1.0, since each row is its own nearest neigbor

print(accuracy_score(y_full, preds_full))
preds_test = model.predict(X_test_copy)
# For some reason this gave some tensorflow deprecation errors???  I did not use tensorflow.



output = pd.DataFrame({'Id': X_test_copy.index,'Cover_type': preds_test})

output.to_csv('submission.csv', index=False)
# Better test to see if the output worked.

output2 = pd.read_csv('submission.csv')

output2.head()