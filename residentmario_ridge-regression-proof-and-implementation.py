import numpy as np
class Ridge:

    def __init__(self, alpha):

        self.alpha = alpha

        

    def fit(self, X, y):

        leftmat = np.linalg.inv(X.T @ X + self.alpha * np.identity(y.shape[1]))

        self.betas = leftmat @ X.T @ y

    

    def predict(self, X):

        return X @ self.betas
X = np.array([[1, 2], [2, 3], [4, 5]])

y = np.array([[1, 7, 9]]).T
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



clf = Ridge(alpha=0.0001)

clf.fit(X, y)

clf.predict(X)



plt.scatter(X[:, 0], y, color='black')

plt.plot(X[:, 0], clf.predict(X))

plt.title('Low-Alpha Ridge on Three Points')