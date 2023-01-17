import numpy as np



class GradientDescentUnivariateLassoRegression:

    def __init__(self, learning_rate=0.01, iterations=1000, alpha=0.5):

        self.learning_rate, self.iterations, self.alpha = learning_rate, iterations, alpha

    

    def fit(self, X, y):

        def soft_threshold(alpha, beta):

            if beta > alpha:

                return beta - alpha

            elif beta < -alpha:

                return beta + alpha

            else:

                return 0

        

        def gradient(X, y, alpha, beta):

            n = len(X)

            ols_term = -2 * np.sum(X*(y - (beta*X))) / n

            soft_term = soft_threshold(alpha, beta) / n

            return ols_term + soft_term

        

        beta = 0.5

        for _ in range(self.iterations):

            grad = gradient(X, y, self.alpha, beta)

            beta = beta - self.learning_rate * grad

            

        self.beta = beta

        

    def predict(self, X):

        return X * self.beta
np.random.seed(42)

X = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.5)

y = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.25)
clf = GradientDescentUnivariateLassoRegression(alpha=0.1)

clf.fit(X, y)



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.scatter(X, y, color='black')

plt.plot(X, clf.predict(X))

plt.gca().set_title("Lasso Regression Linear Regressor")