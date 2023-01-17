import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



import os

print(os.listdir("../input"))



# ignore warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)
train = pd.read_csv("../input/titanicdivethrough-featurecave/feature_cave_train.csv", index_col=0)

test = pd.read_csv("../input/titanicdivethrough-featurecave/feature_cave_test.csv", index_col=0)



# To compare our results in the end, we need the original data

original_train = train.copy()
class MyClassifier:

    

    def __init__(self, n_features):

        self.n_features = n_features

        np.random.seed(0)

        self.w = np.random.normal(loc=0, scale=0.01, size=n_features + 1)

        self.losses = []

    

    def predict(self, x):

        y = sigmoid(np.sum(self.w[:-1]*x, axis=1) + self.w[-1])

        return y

    

    def loss(self, y, t):

        E = - np.sum(t * np.log(y) + (1-t) * np.log(1-y))

        return E

        

    def gradient(self, x, y, t):

        grad = np.zeros(self.w.shape[0])

        for d in range(self.w.shape[0]):

            if d != self.n_features:

                grad[d] = np.sum((y-t)*x[:, d])

            else:

                grad[d] = np.sum((y-t))

        return grad

        

    def update(self, eta, grad):

        w_next = np.zeros(self.w.shape) 

        for d in range(self.w.shape[0]):

            w_next[d] = self.w[d] - eta * grad[d]

        return w_next



    def learn(self, x, t, eta=0.000001, max_steps=100000, tol=0.00001):

        y = self.predict(x)

        for step in range(max_steps):

            error = self.loss(y, t)

            grad = self.gradient(x, y, t)

            self.w = self.update(eta, grad)

            self.losses.append(error)

            y = self.predict(x)

            error_next = self.loss(y, t)

            if (error - error_next) < tol:

                break

                

    def decide(self, y):

        decision = np.zeros(y.shape)

        decision[y >= 0.5] = 1

        decision[y < 0.5] = 0

        return decision.astype(np.int)

    

    def accuracy(self, y, t):

        N = y.shape[0]

        return 1/N * np.sum(1 - np.abs(t-y))

        

    def score(self, x, t):

        y = self.predict(x)

        y = self.decide(y)

        return self.accuracy(y, t)
def sigmoid(x):

    result = 1/(1+np.exp(-x))

    return result
from sklearn.model_selection import train_test_split



features = train.drop(["PassengerId", "Survived"], axis=1).columns



X = train[features].values

Y = train.Survived.values



x_train, x_val, t_train, t_val = train_test_split(X, Y, random_state=0)



train_df = pd.DataFrame(x_train, columns=features)

train_df["Survived"] = t_train

val_df = pd.DataFrame(x_val, columns=features)

val_df["Survived"] = t_val



train_df.head()
one_step_classifier = MyClassifier(n_features = len(features))

predictions = one_step_classifier.predict(x_train)

grad = one_step_classifier.gradient(x_train, predictions, t_train)



features_names = list(features) + ["bias"]



plt.figure(figsize=(20,6))

sns.barplot(x=features_names, y=grad)

plt.ylabel("gradients of loss")

plt.xlabel("per feature parameter $w_{d}$")

plt.title("one step gradients")

plt.xticks(rotation=45);
from sklearn.preprocessing import StandardScaler

# scale the train and test ages



scaler = StandardScaler()

train.Age = scaler.fit_transform(train.Age.values.reshape(-1,1))

test.Age = scaler.transform(test.Age.values.reshape(-1,1))
# visualize the scaled ages distributions of train and test

# hint: you can use sns.distplot of the opensource tool seaborn

plt.figure(figsize=(20,5))

sns.distplot(train.Age.values, label="train", color="darkorange")

sns.distplot(test.Age.values, label="test", color="tomato");

plt.xlabel("Scaled age values");

plt.ylabel("Density")

plt.legend();
assert(np.round(train.Age.mean(),1) == 0)

assert(np.round(train.Age.median(), 1) == -0.1)

assert(np.round(train.Age.min(), 1) == -2.2)

assert(np.round(train.Age.max(), 1) == 3.9)

assert(np.round(test.Age.mean(),1) == 0)

assert(np.round(test.Age.median(), 1) == -0.1)

assert(np.round(test.Age.min(), 1) == -2.2)

assert(np.round(test.Age.max(), 1) == 3.6)
plt.figure(figsize=(20,5))

sns.distplot(train_df.Fare.values, color="Purple")

plt.xlabel("Fare")

plt.ylabel("Density")

plt.title("Fare distribution in train_df");
def get_grad_contributions(y, t, x_d):

    contributions = (y - t) * x_d

    return contributions



fare_contributions = get_grad_contributions(predictions, t_train, train_df.Fare.values)
plt.figure(figsize=(20,5))

sns.distplot(fare_contributions, color="violet")

plt.xlabel("Contribution of fare to the gradient");

plt.ylabel("Density")

plt.title("Do we have extreme contributions to the gradient?");
# create some normally distributed samples:

original = np.random.normal(loc=0, scale=1, size=200)

# add an outlier

shifted = np.array(original.tolist() + [1000])

# comute the mean

print(np.mean(original))

print(np.mean(shifted))
# Just a method to plot our fare distribution with some statistics

def show_fare_distribution():

    plt.figure(figsize=(20,5))

    sns.kdeplot(train[train.Survived==0].Fare, color="Blue", shade=True)

    sns.kdeplot(train[train.Survived==1].Fare, color="Green", shade=True)

    plt.axvline(np.max(train.Fare.values), color="Yellow")

    plt.axvline(np.min(train.Fare.values), color="Yellow")

    plt.axvline(np.mean(train.Fare.values)+np.std(train.Fare.values), color="Orange")

    plt.axvline(np.mean(train.Fare.values)-np.std(train.Fare.values), color="Orange")

    plt.axvline(np.mean(train.Fare.values), color="Red")

    plt.axvline(np.median(train.Fare.values), color="Black")

    plt.xlabel("Fare")

    plt.ylabel("Density")

    return plt
show_fare_distribution()
# perform a log transformation of the fare features in train and test! 

# Hint: You can use pandas apply method, for example: train.Fare = train.Fare.apply(lambda l: some method(l))

train.Fare = train.Fare.apply(lambda l: np.log(l+1))

test.Fare = test.Fare.apply(lambda l: np.log(l+1))

show_fare_distribution()
train.Fare = scaler.fit_transform(train.Fare.values.reshape(-1,1))

test.Fare = scaler.transform(test.Fare.values.reshape(-1,1))
assert(np.round(np.mean(train.Fare), 2) == 0.)

assert(np.round(np.median(train.Fare), 2) == -0.23)

assert(np.round(np.min(train.Fare), 2) == -3.06)

assert(np.round(np.max(train.Fare), 2) == 3.39)

assert(np.round(np.std(train.Fare), 2) == 1.)

assert(np.round(np.mean(test.Fare), 2) == 0.05)

assert(np.round(np.median(test.Fare), 2) == -0.23)

assert(np.round(np.min(test.Fare), 2) == -3.06)

assert(np.round(np.max(test.Fare), 2) == 3.39)

assert(np.round(np.std(test.Fare), 2) == 1.)
sns.countplot(x=train.Sex, hue=train.Survived);
train.Sex = train.Sex.apply(lambda l: np.where(l==0, -1, 1))

test.Sex = test.Sex.apply(lambda l: np.where(l==0, -1, 1))
# Perform this mapping for all remaining categorical features in train and test!



cols_to_use = [col for col in train.columns if col not in ["PassengerId", "Survived", "Age", "Sex", "Fare"]]

for col in cols_to_use:

    train[col] = train[col].apply(lambda l: np.where(l==0, -1, 1))
features = train.drop(["PassengerId", "Survived"], axis=1).columns



X = train[features].values

Y = train.Survived.values



X_old = original_train[features].values

Y_old = original_train.Survived.values



x_train, x_val, t_train, t_val = train_test_split(X, Y, random_state=0)

x_train_old, x_val_old, t_train_old, t_val_old = train_test_split(X_old, Y_old, random_state=0)
new_model = MyClassifier(x_train.shape[1])

new_model.learn(x_train, t_train)

new_losses = new_model.losses

new_score = new_model.score(x_val, t_val)

print(new_score)
old_model = MyClassifier(x_train_old.shape[1])

old_model.learn(x_train_old, t_train_old)

old_losses = old_model.losses

old_score = old_model.score(x_val_old, t_val_old)

print(old_score)
plt.figure()

plt.plot(new_losses, 'g')

plt.plot(old_losses, 'r')

plt.xlabel("Iteration steps")

plt.ylabel("Loss")
assert(np.round(old_score, 4) == 0.7982)

assert(np.round(new_score, 4) == 0.8117)