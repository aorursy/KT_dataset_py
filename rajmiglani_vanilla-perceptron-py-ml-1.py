import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
class Perceptron(object):
    
    """Perceptron classifier.
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
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    
    def __init__(self,eta = 0.01,n_iter=50,random_state=1):
        self.eta = eta;
        self.n_iter = n_iter;
        self.random_state = random_state;
    
    def fit(self,X,Y):
        """Fit training data.
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
        # defining random generator
        rgen = np.random.RandomState(self.random_state);
        # initializing weights randomly from a normal distribution with mean 0 and std 0.01 because
        # we don't want the weights to be all zero for effective learning. Here w_[0] is the bias b
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size = 1 + X.shape[1]);
        
        self.errors_ = [];
        
        # running through the dataset for n_iter epochs
        for _ in range(self.n_iter):
            errors  = 0;
            # applying SGD
            for xi, target in zip(X,Y):
                # getting the delta by calling predict() on a sample i.e. dJ/dwi = (yi - y^i) * xi where J = 0.5 * Sum(yi - y^i)^2 
                # and y^i(or Z) = (wi * Xi + b). Note a linear activation function is used in this implementation. 
                update = self.eta * (target - self.predict(xi));
                # updating weights i.e. w = w + eta * delta(w) * xi
                self.w_[1:] += update * xi; 
                # updating bias b i.e. b = b + eta * delta(w)
                self.w_[0] += update;
                # keeping track of errors while predicting, should be 0 ideally
                errors += int(update != 0.0); 
            # recording errors after each epoch
            self.errors_.append(errors);
        return self;
    
    def net_input(self,X):
        """Calculate net input"""
        # calculating Y^i(or Zi) = W^T * Xi + b, here X is ith training example
        # print(X)
        # print(self.w_[1:])
        return np.dot(X,self.w_[1:]) + self.w_[0];
    
    def predict(self,X):
        """Return class label after unit step"""
        # return correct class label based upon Y^i(or Zi), here X is ith training example
        return np.where(self.net_input(X) >= 0.0,1,-1);
    
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
# get the object 
model = Perceptron(eta = 0.01,n_iter = 10);
# fit the model on our data
model.fit(X,Y);
# plot the errors
plt.plot(range(1,len(model.errors_) + 1),model.errors_,marker='o');
plt.xlabel('Epochs');
plt.ylabel('Number of updates');
plt.show();

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
plot_decision_regions(X, Y, model)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
