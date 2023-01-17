import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import f1_score,roc_auc_score

import matplotlib.pyplot as plt



seed = 1301
path = "../input/titanic/train_and_test2.csv"

data = pd.read_csv(path)



data.head()
###to get train and test

def split_data(data,target,drop,test_size=0.2,seed=seed):

    

    return train_test_split(data.dropna().drop(target+drop,axis=1),

                            data.dropna()[target],

                            test_size=test_size,

                            random_state = seed)
drop,target  = ['Passengerid'],['2urvived']



#split the data

X_train,X_test,y_train,y_test = split_data(data,target,drop=drop)
tcl = DecisionTreeClassifier(max_depth=3,random_state=seed)

tcl.fit(X_train,y_train)

y_pred = tcl.predict(X_test)

y_prob = tcl.predict_proba(X_test)[:,1]

print(f"f1 score: {f1_score(y_test,y_pred):.2f}")

print(f"AUC score: {roc_auc_score(y_test,y_prob):.2f}")
class gradient_booster():

    def __init__(self,loss,gradient_loss,max_depth,nu):

        self.max_depth = max_depth #max depth of all learners

        self.nu = nu #learning rate

        self.loss = loss #loss function to be optimized

        self.gradient_loss = gradient_loss #gradient of the loss function

        self.learners = [] #list with all the learners

        self.loss_history = [] #loss through the process

    

    def fit(self,X,y,epochs):

        base_learner = DecisionTreeClassifier(max_depth=self.max_depth,random_state=seed)

        base_learner.fit(X,y)

        initial_probs = base_learner.predict_proba(X)[:,1]

        

        probs = initial_probs

        preds = initial_probs

        loss_history = [] 

        self.learners = [base_learner]

        target = y

        

        for i in range(epochs):

            target = -gradient_loss(y,probs)

            rt = DecisionTreeRegressor(max_depth=self.max_depth,random_state=seed) #regressor tree

            rt.fit(X,target)

            self.learners.append(rt)

            preds += self.nu*rt.predict(X) #these are not probabilities!!!

            probs = 1 / (1 + np.exp(-preds)) #these are probabilities

            self.loss_history.append(loss(y,probs))

            

        return self

    

    def predict_proba(self,X):

        try:

            preds  = self.learners[0].predict_proba(X)[:,1]

            

            for m in self.learners[1:]:

                preds += self.nu*m.predict(X)



            return 1 / (1 + np.exp(-preds))

        

        except NotFittedError:

            print("Model not fitted yet")
def loss(y,p):

    return -2.0 * np.mean(y * np.log(p/(1-p)) - np.logaddexp(0.0, np.log(p/(1-p)))) 
def gradient_loss(y,p):

    return (p-y)/(p*(1-p))
booster = gradient_booster(loss=loss,gradient_loss=gradient_loss,max_depth=3,nu=0.01)

X, y, epochs = X_train.values, y_train.values.ravel(), 50

booster.fit(X,y,epochs)

y_prob = booster.predict_proba(X_test)

y_pred = 1*(y_prob>0.5)

print(f"f1 score: {f1_score(y_test,y_pred):.2f}")

print(f"AUC score: {roc_auc_score(y_test,y_prob):.2f}")
fig,ax = plt.subplots(1,1,figsize=(6,6))

plt.plot(range(epochs),booster.loss_history)

plt.show()