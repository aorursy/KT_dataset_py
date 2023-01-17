import warnings

warnings.filterwarnings("ignore")



import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston

from prettytable import PrettyTable

from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error
#Implementing my own version of SGD, will be calling it Naive SGD...



class NaiveSGD():

    def __init__(self, X, y, max_iter=1500, r=.001):

        self.X = X

        self.y = y

        self.max_iter = max_iter

        self.r = r

    

    def getK_Sum(self, w_prev, b):

        K = np.random.randint(0, self.X.shape[0]-25)



        sum_Kw = np.zeros((1,self.X.shape[1]))

        sum_Kb = 0

        

        for i in range(K, K+25):

            sum_Kw += 2*self.X[i]*(self.y[i] - (np.dot(w_prev, self.X[i]) + b))

            sum_Kb += 2*(self.y[i] - (np.dot(w_prev, self.X[i]) + b))

            

        return sum_Kw/25, sum_Kb/25

    

    def fit(self):

        r = self.r

        w = np.zeros((1, self.X.shape[1]))

        b = 0

        

        for i in range(self.max_iter):

            w_, b_ = self.getK_Sum(w, b)

            w += r*w_

            b += r*b_

            #r /= 2



        self.optimal_w = w

        self.optimal_b = b

        

    def predict(self, X):

        y_pred = []

        

        for X_i in X:

            y_pred.append(self.optimal_b + np.dot(self.optimal_w, X_i))

        

        return np.array(y_pred)
def plotChart(y_test, y_pred):

    plt.scatter(y_test, y_pred)

    plt.xlabel("Prices: $Y_i$")

    plt.ylabel("Predicted prices: $\hat{Y}_i$")

    plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

    plt.show()
X = load_boston().data

Y = load_boston().target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)



scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
nsgd = NaiveSGD(X_train,y_train)

nsgd.fit()
y_spred = nsgd.predict(X_test)

plotChart(y_test, y_spred)
clf = SGDRegressor(loss='squared_loss')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

plotChart(y_test, y_pred)
table = PrettyTable()

table.field_names = ['Dimension','Naive SGD', 'SKLEARN SGD']



sk_w = clf.coef_

for i, wi in enumerate(sk_w):

    table.add_row([i+1, nsgd.optimal_w[0][i], wi])



print (table)
table = PrettyTable()

table.field_names = ["Model's Name", "MSE value"]



table.add_row(["Naive's SGD", mean_squared_error(y_test,y_pred)])

table.add_row(["SKLEARN's SGD", mean_squared_error(y_test,y_spred)])



print (table)