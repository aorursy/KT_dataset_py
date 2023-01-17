import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.datasets import load_iris

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt

import plotly.graph_objects as go
iris = load_iris()

df = pd.DataFrame(iris.data)

df[4] = iris.target

df.head()
# select setosa and versicolor

y = df.iloc[0:100, 4].values

y = np.where(y == 0, -1, 1)



# extract sepal length and petal length

X = df.iloc[0:100, [0, 2]].values



# plot data

plt.scatter(X[:50, 0], X[:50, 1],

            color='red', marker='o', label='setosa')

plt.scatter(X[50:100, 0], X[50:100, 1],

            color='blue', marker='x', label='versicolor')



plt.xlabel('sepal length [cm]')

plt.ylabel('petal length [cm]')

plt.legend(loc='upper left')



# plt.savefig('images/02_06.png', dpi=300)

plt.show()
px.scatter(df[df[4]!=1], x=0, y=2, color=4, color_continuous_scale=[[0.0, "rgb(165,0,38)"],

                [0.1111111111111111, "rgb(215,48,39)"],

                [1.0, "rgb(49,54,149)"]])
class Perceptron(object):

    """Perceptron classifier.



    Parameters

    ------------

    eta : float

      Learning rate (between 0.0 and 1.0)

    n_iter : int

      Passes over the training dataset.

    random_state : int

      Random number generator seed for random weight

      initialization.



    Attributes

    -----------

    w_ : 1d-array

      Weights after fitting.

    errors_ : list

      Number of misclassifications (updates) in each epoch.



    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):

        self.eta = eta

        self.n_iter = n_iter

        self.random_state = random_state



    def fit(self, X, y):

        """Fit training data.



        Parameters

        ----------

        X : {array-like}, shape = [n_samples, n_features]

          Training vectors, where n_samples is the number of samples and

          n_features is the number of features.

        y : array-like, shape = [n_samples]

          Target values.



        Returns

        -------

        self : object



        """

        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []



        for _ in range(self.n_iter):

            errors = 0

            for xi, target in zip(X, y):

                update = self.eta * (target - self.predict(xi))

                self.w_[1:] += update * xi

                self.w_[0] += update

                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self



    def net_input(self, X):

        """Calculate net input"""

        return np.dot(X, self.w_[1:]) + self.w_[0]



    def predict(self, X):

        """Return class label after unit step"""

        return np.where(self.net_input(X) >= 0.0, 1, -1)
ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)
fig = go.Figure(data=go.Scatter(x=np.array(range(1, len(ppn.errors_)+1)), y=ppn.errors_))

fig.show()
# coefficients

ppn.w_
from matplotlib.colors import ListedColormap



def plot_decision_regions(X, y, classifier, resolution=0.02):



    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    # plot class samples

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], 

                    y=X[y == cl, 1],

                    alpha=0.8, 

                    c=colors[idx],

                    marker=markers[idx], 

                    label=cl, 

                    edgecolor='black')

plot_decision_regions(X, y, classifier=ppn)
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),

                           np.arange(x2_min, x2_max, 0.02))

Z = ppn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

Z = Z.reshape(xx1.shape)





fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=X[:50,0], y=X[:50,1],

                    mode='markers',

                    name='1'))

fig.add_trace(go.Scatter(x=X[50:100,0], y=X[50:100,1],

                    mode='markers',

                    name='-1'))



fig.add_trace(

    go.Contour( x= np.arange(x1_min, x1_max, 0.02),

                y= np.arange(x2_min, x2_max, 0.02),

        z=Z,

        colorscale = [[0.0, "yellow"],[1.0, "lightgreen"]],

               opacity = 0.5

    ))

fig.show()

class AdalineGD(object):

    """ADAptive LInear NEuron classifier.



    Parameters

    ------------

    eta : float

      Learning rate (between 0.0 and 1.0)

    n_iter : int

      Passes over the training dataset.

    random_state : int

      Random number generator seed for random weight

      initialization.





    Attributes

    -----------

    w_ : 1d-array

      Weights after fitting.

    cost_ : list

      Sum-of-squares cost function value in each epoch.



    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):

        self.eta = eta

        self.n_iter = n_iter

        self.random_state = random_state



    def fit(self, X, y):

        """ Fit training data.



        Parameters

        ----------

        X : {array-like}, shape = [n_samples, n_features]

          Training vectors, where n_samples is the number of samples and

          n_features is the number of features.

        y : array-like, shape = [n_samples]

          Target values.



        Returns

        -------

        self : object



        """

        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.cost_ = []



        for i in range(self.n_iter):

            net_input = self.net_input(X)

            # Please note that the "activation" method has no effect

            # in the code since it is simply an identity function. We

            # could write `output = self.net_input(X)` directly instead.

            # The purpose of the activation is more conceptual, i.e.,  

            # in the case of logistic regression (as we will see later), 

            # we could change it to

            # a sigmoid function to implement a logistic regression classifier.

            output = self.activation(net_input)

            errors = (y - output)

            self.w_[1:] += self.eta * X.T.dot(errors)

            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0

            self.cost_.append(cost)

        return self



    def net_input(self, X):

        """Calculate net input"""

        return np.dot(X, self.w_[1:]) + self.w_[0]



    def activation(self, X):

        """Compute linear activation"""

        return X



    def predict(self, X):

        """Return class label after unit step"""

        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
fig, ax =plt.subplots(nrows = 1, ncols=2,   figsize = (10,4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)

ax[0].plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_), marker = 'o')

ax[0].set_xlabel('Epochs')

ax[0].set_ylabel('log(Sum-squared-error)')



ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)

ax[1].plot(range(1,len(ada2.cost_)+1), np.log10(ada2.cost_), marker = 'o')

ax[1].set_xlabel('Epochs')

ax[1].set_ylabel('log(Sum-squared-error)')
import ipywidgets as widgets

from IPython.display import clear_output

int_range = widgets.FloatSlider(

    value = 0.001,

    min=0.000001,

    max=0.001,

    step=0.00001,

    description='Test:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='.6f',

)

output2 = widgets.Output()







def on_value_change(change):

    with output2:

        #print(change['new'])

        plt.clf()

        fig=plt.figure(figsize = (4,4))

        ada1 = AdalineGD(n_iter=10, eta=change['new']).fit(X,y)

        plt.plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_), marker = 'o')

        plt.xlabel('Epochs')

        plt.ylabel('log(Sum-squared-error)')

        plt.show()

        clear_output(wait = True)

        

    

display(int_range, output2)



int_range.observe(on_value_change, names='value')
X_std = np.copy(X)

X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()

X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
ada = AdalineGD(n_iter = 15, eta = 0.01)

ada.fit(X_std,y)



plot_decision_regions(X_std, y, classifier=ada)

plt.tight_layout()

plt.show()



plt.plot(range(1,len(ada.cost_)+1),ada.cost_, marker = 'o')

plt.tight_layout()

plt.show()
class AdalineSGD(object):

    """ADAptive LInear NEuron classifier.



    Parameters

    ------------

    eta : float

      Learning rate (between 0.0 and 1.0)

    n_iter : int

      Passes over the training dataset.

    shuffle : bool (default: True)

      Shuffles training data every epoch if True to prevent cycles.

    random_state : int

      Random number generator seed for random weight

      initialization.





    Attributes

    -----------

    w_ : 1d-array

      Weights after fitting.

    cost_ : list

      Sum-of-squares cost function value averaged over all

      training samples in each epoch.



        

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):

        self.eta = eta

        self.n_iter = n_iter

        self.w_initialized = False

        self.shuffle = shuffle

        self.random_state = random_state

        

    def fit(self, X, y):

        """ Fit training data.



        Parameters

        ----------

        X : {array-like}, shape = [n_samples, n_features]

          Training vectors, where n_samples is the number of samples and

          n_features is the number of features.

        y : array-like, shape = [n_samples]

          Target values.



        Returns

        -------

        self : object



        """

        self._initialize_weights(X.shape[1])

        self.cost_ = []

        for i in range(self.n_iter):

            if self.shuffle:

                X, y = self._shuffle(X, y)

            cost = []

            for xi, target in zip(X, y):

                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)

            self.cost_.append(avg_cost)

        return self



    def partial_fit(self, X, y):

        """Fit training data without reinitializing the weights"""

        if not self.w_initialized:

            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:

            for xi, target in zip(X, y):

                self._update_weights(xi, target)

        else:

            self._update_weights(X, y)

        return self



    def _shuffle(self, X, y):

        """Shuffle training data"""

        r = self.rgen.permutation(len(y))

        return X[r], y[r]

    

    def _initialize_weights(self, m):

        """Initialize weights to small random numbers"""

        self.rgen = np.random.RandomState(self.random_state)

        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)

        self.w_initialized = True

        

    def _update_weights(self, xi, target):

        """Apply Adaline learning rule to update the weights"""

        output = self.activation(self.net_input(xi))

        error = (target - output)

        self.w_[1:] += self.eta * xi.dot(error)

        self.w_[0] += self.eta * error

        cost = 0.5 * error**2

        return cost

    

    def net_input(self, X):

        """Calculate net input"""

        return np.dot(X, self.w_[1:]) + self.w_[0]



    def activation(self, X):

        """Compute linear activation"""

        return X



    def predict(self, X):

        """Return class label after unit step"""

        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)

ada.fit(X_std, y)



plot_decision_regions(X_std, y, classifier=ada)

plt.title('Adaline - Stochastic Gradient Descent')

plt.xlabel('sepal length [standardized]')

plt.ylabel('petal length [standardized]')

plt.legend(loc='upper left')



plt.tight_layout()

# plt.savefig('images/02_15_1.png', dpi=300)

plt.show()



plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')

plt.xlabel('Epochs')

plt.ylabel('Average Cost')



plt.tight_layout()

# plt.savefig('images/02_15_2.png', dpi=300)

plt.show()