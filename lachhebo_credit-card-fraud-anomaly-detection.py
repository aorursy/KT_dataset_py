import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import SimpSOM as sps # Kohohen maps

from sklearn.preprocessing import minmax_scale
df = pd.read_csv("../input/creditcard.csv")

df_array =  df.values

df_normalised = minmax_scale(df_array)
net = sps.somNet(20, 20, df_normalised, PBC=True)

net.train(0.01, 10000)

net.nodes_graph(colnum=30)

net.diff_graph()
from sklearn.model_selection import train_test_split, KFold
X = df_array[:,0:29]

y = df_array[:,30]



X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify= y)
from sklearn.ensemble import IsolationForest

from sklearn.metrics  import average_precision_score

import matplotlib.pyplot as plt
kf = KFold(n_splits=3, random_state=42, shuffle=False)
resultats = {}



for t in  [50,70,80,100,120,140]:

    for psi in [250,300,400,600,800,1200,2000,5000]:

        clf = IsolationForest(n_estimators = t, max_samples= psi, behaviour = "new", contamination = "auto", random_state = 42)

        result = []

        for train_index, test_index in kf.split(X_train):

            clf.fit(X_train[train_index])

            

            score = abs(clf.score_samples(X_train[test_index]))

            

            result.append(average_precision_score(y_train[test_index],score))

            

        resultats["t_"+str(t)+"psi_"+str(psi)] = {

            "t": t,

            "psi": psi,

            "score": np.mean(result)

        }
## Let's find out which is the best model we trained



score_max, ind_best = (0,None)



for model in resultats :

    if resultats[model]["score"] > score_max :

        score_max  = resultats[model]["score"]

        ind_best = model

        
## And the best model is .. : 



print("t : ", resultats[ind_best]["t"], "psi : ", resultats[ind_best]["psi"], "score :",resultats[ind_best]["score"])



ifor = IsolationForest(n_estimators = resultats[ind_best]["t"], max_samples= resultats[ind_best]["psi"], behaviour = "new",

                contamination = "auto")



ifor.fit(X_train)
score  = abs(ifor.score_samples(X_test)) 
from sklearn.metrics import precision_score, recall_score

from matplotlib import pyplot as plt



score  = abs(ifor.score_samples(X_test))



def precisifun(i,X_test,y_test):

    ind = np.argpartition(score, -int((i/100)*len(X_test)))[-int((i/100)*len(X_test)):]

    pred = np.zeros(len(y_test))

    pred[ind] = 1

    return precision_score(y_test, pred)

    

def recallfun(i,X_test,y_test):

    ind = np.argpartition(score, -int((i/100)*len(X_test)))[-int((i/100)*len(X_test)):]

    pred = np.zeros(len(y_test))

    pred[ind] = 1

    return recall_score(y_test, pred)

    

precision = []

recall = []

for i in np.concatenate( ([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],range(1,100))):

    precision.append(precisifun(i,X_test,y_test))

    recall.append(recallfun(i,X_test,y_test))

    

plt.plot(recall, precision)

def precisifun(i,X_test,y_test):

    ind = np.argpartition(score, -int((i/100)*len(X_test)))[-int((i/100)*len(X_test)):]

    pred = np.zeros(len(y_test))

    pred[ind] = 1

    return precision_score(y_test, pred)

    

def recallfun(i,X_test,y_test):

    ind = np.argpartition(score, -int((i/100)*len(X_test)))[-int((i/100)*len(X_test)):]

    pred = np.zeros(len(y_test))

    pred[ind] = 1

    return recall_score(y_test, pred)

    

precision = []

recall = []

for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

    precision.append(precisifun(i,X_test,y_test))

    recall.append(recallfun(i,X_test,y_test))

    

plt.plot(recall, precision)