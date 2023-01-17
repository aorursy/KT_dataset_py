import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
class LogisticRegression(object):
    """Logistic Regression Classifier using gradient descent.
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.
    """
    
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta;
        self.n_iter = n_iter;
        self.random_state = random_state;
        
    def fit(self,X,Y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state);
        self.w_ = rgen.normal(loc=0.0, scale=0.01,size=1 + X.shape[1]);
        self.cost_ = [];
        
        # run through the epochs
        for i in range(self.n_iter):
            # find Z =  w^T * X
            net_input = self.net_input(X);
            # find Y^ = activation(Z)
            output = self.activation(net_input);
            # find the error (Y - Y^)
            errors = (Y - output);
            # updating weigths i.e. w = w + alpha * dJ/dw = w + aplha * (Y - Y^) * X, here alpha ie the learning rate
            self.w_[1:] += self.eta * X.T.dot(errors);
            # updating bias i.e b = b + alpha * sum(Y - Y^), here Y and Y^ are matrices hence summing all elements
            self.w_[0] += self.eta * errors.sum();
            # compute the logistic `cost` instead of the sum of squared errors cost as in Perceptron and Adaline
            cost = (-Y.dot(np.log(output)) -((1 - Y).dot(np.log(1 - output))))
            # record cost after each epoch
            self.cost_.append(cost)
        return self;
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0];
                                                
    def activation(self, Z):
        """Compute logistic sigmoid activation"""
        # clip Z
        return 1. / (1. + np.exp(-np.clip(Z, -250, 250)));
    
    def predict(self,X):
        """Return class label after unit step"""
        # here we threshold on Z for better understanding and less calculations, we may also threshold on activation(Z) with threshold 0.5
        return np.where(self.net_input(X) >= 0.0, 1, 0);
                                                
df = pd.read_csv('../input/Iris.csv');
df.tail()
# select setosa and versicolor samples
# an overview of first 100 samples
df.iloc[0:100,5].value_counts();
# get the class labels for first 100 samples
Y = df.iloc[0:100,5].values;
# set class labels as 1 and -1
Y = np.where(Y == 'Iris-setosa',0,1);
# select sepal and petal length freatures
X = df.iloc[0:100,[1,3]].values

# plot the data for setosa class
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
# plot the data for versicolor class
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='vesicolor')
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()

# making a copy of the dataset
X_std = np.copy(X)
# standardizing sepal length
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
# standardizing petal length
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
def plot_decision_regions(X,Y,model,resolution = 0.02):
    # setup colormap and markers
    markers = ('s','x','o','^','v');
    colors = ('red','blue','lightgreen','gray','cyan');
    cmap = ListedColormap(colors[:len(np.unique(Y))]);
    # plot the decision surface
    # determin minimim ans maximum values for the two features
    x1_min,x1_max =  X[:,0].min() - 1, X[:,0].max() + 1;
    x2_min,x2_max =  X[:,1].min() - 1, X[:,1].max() + 1; 
    # create a pair of grid arrays
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    # predict on our model by flattening the arrays and making it into a matrix
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cl, 0],y=X[Y == cl, 1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')
# get the model object
lgr = LogisticRegression(eta = 0.05,n_iter = 100, random_state = 1);
# fit the model
lgr.fit(X_std,Y);
# plot the regions
plot_decision_regions(X=X_std,Y=Y,model=lgr);
plt.xlabel('petal length [standardized]');
plt.ylabel('petal width [standardized]');
plt.legend(loc='upper left');
plt.show();