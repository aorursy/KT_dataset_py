import numpy as np

from sklearn.datasets import make_blobs

 

X_all, y_all = make_blobs(

    n_samples=200,

    n_features=2,

    centers=2, #classes

    random_state=123

)

 

 

#Note: We are not adding bias column as webwont be using vectors.

 

 

# plot

import matplotlib.pyplot as plt

plt.scatter(x=X_all[:, 0], y=X_all[:, 1], c=y_all,

            cmap=plt.cm.Paired)

plt.show();

 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)

 

print(X_train.shape)

print(y_train.shape)

print(X_test.shape, y_test.shape)
#TRAINING (difft than ususal)

#1. check if y(z_i) > k or < k

#2. calculate mistakes accordingly. 

#3. Correct them until convergence



#PREDICTING

#3. predict using sign of z_i.

#      y(z_i) cant be used for prediction



#Note

#We are not using vectors
class SVM:

    def __init__(self, lr=1e-2, epochs=1000, lambda_val=1e-2, k=1):

        self.lr = lr

        self.epochs = epochs

        self.lambda_val = lambda_val

        self.k = k

        self.w = None

        self.b = None

        

    def fit(self, X_train, y_train):

        n_feats = X_train.shape[1]

        self.w = np.ones(n_feats)

        self.b = np.ones(1)

        # make sure class labels are -1 and +1

        y_train = np.where(y_train==0, -1, 1)

        

        for e in range(self.epochs):

            for i, x_i in enumerate(X_train):

                #1. predict

                k_hat = y_train[i] * (np.dot(x_i, self.w) + self.b)

                correct_classification = k_hat > self.k

                #2. find mistakes

                if correct_classification:

                    dw = 2 * self.lambda_val * self.w

                    db = 0

                else:

                    dw = (2 * self.lambda_val * self.w) - (y_train[i] * x_i) #lazy to change wrong eqn above in latex

                    db = (-1) * (y_train[i])

                #3. correct

                self.w = self.w - self.lr * dw

                self.b = self.b - self.lr * db

        

    def predict(self, X_test):

        return [self._predict(x_q) for x_q in X_test]

        

    def _predict(self, x_q):

        pred = np.dot(x_q, self.w) + self.b

        return 0 if pred<0 else 1
clf = SVM()

clf.fit(X_train, y_train)

y_test_preds = clf.predict(X_test)



acc = np.sum(y_test_preds == y_test) / len(y_test)

print(acc) 
# Plot results



w = clf.w

b = clf.b

k = clf.k



x_min, x_max = X_train[:,0].min(), X_train[:, 0].max()

xs = np.linspace(x_min, x_max, 3)



# Decision boundary

# wx + b = 0

# => w[0]x1 + w[1]x2 + b = 0

# => y = - (w[1]/w[0])x - (b/w[0])

ys = -1 * (1/w[0]) * (w[1]*xs + b)

plt.plot(xs, ys)



# SVs

# y = mx + b +- (k/w^2)

dist = k / np.sqrt(np.sum(w**2))

sv1 = ys + dist

sv2 = ys - dist

plt.plot(xs, sv1, color='r')

plt.plot(xs, sv2, color='r')



# x_test

plt.scatter(x=X_test[:,0], y=X_test[:,1], c=y_test)

#plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train, cmap=plt.cm.Paired)



#Note: from fig. train acc is less than 1.0 but test acc is 1.0 