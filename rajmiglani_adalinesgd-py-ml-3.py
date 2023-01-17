import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
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
        Random number generator seed for random weight initialization.
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value averaged over all training samples in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=10,shuffle=True, random_state=None):
        self.eta = eta;
        self.n_iter = n_iter;
        self.w_initialized = False;
        self.shuffle = shuffle;
        self.random_state = random_state;
    
    def fit(self, X, Y):
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
        # initialize weights
        self._initialize_weights(X.shape[1]);
        self.cost_ = [];

        # run through the epochs
        for _ in range(self.n_iter):
            # shuffle the dataset
            if self.shuffle:
                X,Y = self._shuffle(X,Y);
            cost = [];
            # run through each training example
            for xi, target in zip(X,Y):
                # append cost after each update
                cost.append(self._update_weights(xi,target));
            # average the cost after each epoch
            avg_cost = sum(cost) / len(Y);
            # record the average cost after each epoch
            self.cost_.append(avg_cost);
        return self;
    
    def partial_fit(self,X,Y):
        """Fit training data without reinitializing the weights for online data"""
        # initialize weights if not initialized
        if not self.w_initialized:
            self._initialezed_weights(X.shape[1]);
        if Y.ravel().shape[0] > 1:
            for xi, target in zip(X,Y):
                self._update_weights(xi,target);
        else:
            self._update_weights(X,Y);
        return self;
    
    def _shuffle(self,X,Y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(Y));
        return X[r], Y[r];
    
    def _initialize_weights(self,m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state);
        # size 1 + n_features, here is for bias,b 
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m);
        self.w_initialized = True;
        
    def _update_weights(self,xi,target):
        """Apply Adaline learning rule to update the weights"""
        # get output for a training example i.e. activation(Z), where Z = w^T * X + b
        output = self.activation(self.net_input(xi));
        error = (target - output);
        # updating weigths i.e. w = w + alpha * dJ/dw = w + aplha * (Yi - Y^i) * Xi, here alpha ie the learning rate
        self.w_[1:] += self.eta * xi.dot(error);  
        # updating bias i.e b = b + alpha * (Yi - Y^i)
        self.w_[0] += self.eta * error;
        # calculating cost i.e. J = 1/2 * (Yi - Y^i)^2
        cost = 1/2 * error**2;
        return cost;
    
    def net_input(self, X):
        """Calculate net input"""
        # calcu;ating Z = w^T * xi + b
        return np.dot(X, self.w_[1:]) + self.w_[0];
    
    def activation(self, X):
        """Compute linear activation"""
        return X;
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1);
        
df = pd.read_csv('../input/Iris.csv');
df.tail()
# select setosa and versicolor samples
# an overview of first 100 samples
df.iloc[0:100,5].value_counts();
# get the class labels for first 100 samples
Y = df.iloc[0:100,5].values;
# set class labels as 1 and -1
Y = np.where(Y == 'Iris-setosa',-1,1);
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

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, Y)
plot_decision_regions(X_std, Y, model=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()