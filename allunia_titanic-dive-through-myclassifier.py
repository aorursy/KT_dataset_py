import os
print(os.listdir("../input"))
print(os.listdir("../input/titanicdivethrough"))
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
train = pd.read_csv("../input/titanicdivethrough/prepared_train.csv", index_col=0)
train.head()
test = pd.read_csv("../input/titanicdivethrough/prepared_test.csv", index_col=0)
test.head()
class MyClassifier:
    
    def __init__(self):
        np.random.seed(0)
        self.w = np.random.normal(loc=0, scale=0.01, size=7)
        self.losses = []
        pass
    
    def predict(self, x):
        pass
    
    def loss(self, y, t):
        pass
    
    def gradient(self, x, y, t):
        pass
    
    def update(self, eta, grad):
        pass
    
    def learn(self, x, t, eta, max_steps, tol):
        pass
    
    def score(self, x, t):
        pass
# It's your turn: Delete the pass and implement the sigmoid function:
def sigmoid(x):
    pass
    #result = <your code>
    #return result
#assert(sigmoid(0) == 0.5)
#assert(np.round(sigmoid(-6), 2) == 0.0)
#assert(np.round(sigmoid(6), 2) == 1.0)
# Your task: Delete the pass and fill in your solution. Use w in your equation as parameters.
def predict(x, w):
    pass
    #y = <your code>
    #return y
np.random.seed(0)
w = np.random.normal(loc=0, scale=0.01, size=7)

X = train.drop(["PassengerId", "Survived"], axis=1).values
# Y = predict(X, w)
plt.figure(figsize=(20,5))
#sns.distplot(Y)
plt.xlabel("predicted values y")
plt.ylabel("frequency")
plt.title("Initial predictions of all train passengers")
#assert(np.round(np.mean(Y), 2) == 0.51)
#assert(np.round(np.std(Y), 2) == 0.09)
#assert(np.round(np.median(Y), 2) == 0.54)
#assert(np.round(np.min(Y), 2) == 0.01)
#assert(np.round(np.max(Y), 2) == 0.67)
def are_they_close(y, t):
    pass
    #p = <your code>
    #return p
t = train.Survived.values[0]
#y = Y[0]
#probability = are_they_close(y,t)
#probability
#assert(np.round(are_they_close(y,t), 2) == 0.45)
# Fill in the code to compute the loss of N passengers. The method arguments y and t are both vectors of length N. 
def loss(y, t):
    pass
    # E = <your code>
    # return E
prediction = 0.5 * np.ones(train.shape[0])
target = train.Survived.values
#assert(np.round(loss(prediction, target))==618.0)
# implement the equation for the gradient of E with respect to parameters self.w
def gradient(x, y, t):
    #grad = np.zeros(w.shape[0])
    #for d in range(w.shape[0]):
        #grad[d] = <your code>
    #return grad
    pass
    
# implement the update equation for all parameters w in self.w
def update(w, eta, grad):
    # w_next = np.zeros(w.shape) 
    #for d in range(w.shape[0]):
        #w_next[d] = <your code>
    #return w_next
    pass
T = train.Survived.values
#grads = gradient(X, Y, T)
#w_next = update(w, 0.5 grads)
grads_control = np.array([429, -80, 3973, 59, 0, -5587, 10])
new_weights_control = np.array([-215, 40, -1987, -30, 0, 2793, -5])
#np.testing.assert_array_almost_equal(grads_control, grads, 0)
#np.testing.assert_array_almost_equal(new_weights_control, w_next, 0)
#dw = w_next - w 
fig, ax = plt.subplots(1,2, figsize=(20,5))
#sns.barplot(x=train.drop(["PassengerId", "Survived"], axis=1).columns, y=grads, ax=ax[0])
#sns.barplot(x=train.drop(["PassengerId", "Survived"], axis=1).columns, y=dw, ax=ax[1])
ax[0].set_ylabel("gradients of loss")
ax[0].set_xlabel("per feature weight")
ax[0].set_title("Gradients of 1 step")
ax[1].set_ylabel("change of weights")
ax[1].set_xlabel("per feature")
ax[1].set_title("Change of weights after 1 step")
#contributions_age = (Y-T) * train.Age.values
#contributions_fare = (Y-T) * train.Fare.values

fig, ax = plt.subplots(1,2,figsize=(20,5))
#sns.distplot(contributions_age, ax=ax[0], color="Orange")
ax[0].set_title("Contributions to the age gradient")
ax[0].set_xlabel("contribution $(y_{n} - t_{n}) \cdot age $")
#sns.distplot(contributions_fare, ax=ax[1], color="Purple")
ax[1].set_title("Contributions to the fare gradient")
ax[1].set_xlabel("contribution $(y_{n} - t_{n}) \cdot fare $")
plt.figure(figsize=(10,3))
sns.distplot(train.Fare)
plt.title("Remember the fare outliers")
def learn(x, t, eta, max_steps, tol):
    losses = []
    np.random.seed(0)
    w = np.random.normal(loc=0, scale=0.01, size=7)
    y = predict(x, w)
    #for step in range(max_steps):
        #current_loss = <your code>
        #losses.append(current_loss)
        #grads = <your code>
        #w = <your code>
        #y = <your code>
        #next_loss = <your code>
        #if (current_loss - next_loss) < tol:
            #break
    #return losses, w
    pass

X = train.drop(["PassengerId", "Survived"], axis=1).values
T = train.Survived.values

#losses, w = learn(X, T, 0.000001, 100000, 0.00001)
#w
#assert(np.round(np.max(losses)) == 703)
#assert(np.round(np.mean(losses)) == 422)
#assert(np.round(np.min(losses)) == 404)
#assert(np.round(np.std(losses)) == 30)
#weights_control = np.array([-0.562, 2.58, -0.015, -0.284, -0.092, 0.008, 0.246])
#np.testing.assert_array_equal(np.round(w, 3), weights_control, 3)
plt.figure(figsize=(15,5))
#plt.plot(losses)
plt.xlabel("Iteration steps")
plt.ylabel("cross-entropy loss")
# Implement the accuracy score. Hint use np.sum and np.abs
def score(x, t):
    N = x.shape[0]
    y = predict(x, w)
    #accuracy = <your code>
    #return accuracy
    pass
#accuracy = score(X, T)
#accuracy
#assert(np.round(accuracy, 2) == 0.80)
features = train.drop(["PassengerId", "Survived"], axis=1).columns

x_train = train[features].values
t_train = train.Survived.values

classifier = MyClassifier()
#classifier.learn(x_train, t_train, 0.0000001, 100000, 0.00001)
#score = classifier.score(x_train, t_train)
#score
plt.figure(figsize=(15,5))
plt.plot(classifier.losses)
plt.xlabel("Iteration steps")
plt.ylabel("Loss")
plt.title("My classifer losses")
#assert(np.round(score, 2) == 0.80)
#assert(np.round(np.max(classifier.losses)) == 703)
#assert(np.round(np.mean(classifier.losses)) == 422)
#assert(np.round(np.min(classifier.losses)) == 404)
#assert(np.round(np.std(classifier.losses)) == 30)
#weights_control = np.array([-0.562, 2.58, -0.015, -0.284, -0.092, 0.008, 0.246])
#np.testing.assert_array_equal(np.round(classifier.w, 3), weights_control, 3)