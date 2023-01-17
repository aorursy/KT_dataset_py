# import dependancies
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

import matplotlib.pylab as plt
import seaborn as sns;
sns.set_context('poster')
sns.set_style('darkgrid')
warnings.simplefilter("ignore")

# Read in data and drop nans
df = pd.read_csv('../input/kc_house_data.csv')
df.dropna(axis=0, how='any', inplace=True)
# Drop date and zipcode as we will not use them in the analysis
df.drop(labels=['date','zipcode'],axis=1, inplace=True)
# Create development dataset "df" and test dataset "df_test" 
df, df_test = train_test_split(df, test_size=0.1)
df.reset_index(drop=True,inplace=True)
df_test.reset_index(drop=True,inplace=True)

# Create a matrix from the columns of the development dataset
# and scale these columns 
# Apply the same scaling to the test set
cols = [i for i in list(df.columns) if i not in ['id','price']]
x_dev = df.as_matrix(columns=cols)
x_test = df_test.as_matrix(columns=cols)
# Fit scaler only on dev set only
scaler = StandardScaler().fit(x_dev)
df.drop(labels=cols,axis=1,inplace=True)
df_test.drop(labels=cols,axis=1,inplace=True)
for col in cols:
    df[col] = scaler.fit_transform(x_dev[:,cols.index(col)].reshape(-1,1))
    df_test[col] = scaler.fit_transform(x_test[:,cols.index(col)].reshape(-1,1))
    

df.head()
df.shape
for col in cols:
    fig, ax = plt.subplots(figsize=(12,8))
    df.plot(kind='scatter', x=col, y='price', ax=ax, s=10, alpha=0.5)
    print("corr between price and", col,":",pearsonr(df[col].values, df.price.values)[0])
    plt.show()
x_check_cov = df.as_matrix(columns=cols)
x_check_cov.shape
Sigma = (1/x_check_cov.shape[0])*np.dot(x_check_cov.T,x_check_cov)
Sigma = abs(Sigma)
np.tri
mask = np.zeros_like(Sigma)
mask[np.triu_indices_from(Sigma)] = True
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(15,10))
    sns.heatmap(Sigma, mask=mask, square=True, vmin=0.4, ax=ax, linewidths=1, xticklabels=cols, yticklabels=cols)
    plt.show()
discard_cols = ['grade','bathrooms','sqft_above','sqft_lot','sqft_lot15']
cols = [i for i in list(df.columns) if i not in discard_cols]
df.drop(labels=discard_cols,axis=1,inplace=True)
df_test.drop(labels=discard_cols,axis=1,inplace=True)
df.head()
class RidgeGD:
    """
    Performs ridge regression on a dataset with L2 regularization (from scratch implementation)
    """
    
    def __init__(self,lambda_=1,alpha=0.01, batchSize=32,n_iter=100,output_every=10):
        self.lambda_        = lambda_ # the penalty / regularization size
        self.alpha          = alpha # the learning rate
        self.batchSize      = batchSize # the size mini batch for gradient descent
        self.n_iter         = n_iter # the numner of iterations in mini batch gradient descent
        self.output_every   = output_every # how often to print error
        
    def cost(self,x,y,w):
        """
        Calculate the cost with current weights
        INPUT: data x, targets y, weights w
        OUTPUT: cost
        """
        # X ~ mxn
        # y ~ mx1
        # W ~ nx1
        m = x.shape[0]
        h = np.dot(x,w) # mxn * nx1 = mx1
        error = h - y
        w[0] = 0 # dont regularize bias
        J = (1/(2*m))*np.sum(error**2) + (self.lambda_/(2*m))*np.sum(w**2)
        return J
     
    def grad(self,x,w,y):
        """
        Calculate the gradient of the cost function
        INPUT: data x, targets y, weights w
        OUTPUT: gradient of cost function
        """
        m = x.shape[0]
        h = np.dot(x,w)
        error = h - y # mx1
        w[0] = 0 # dont regularize bias term
        partial = (1/m)*np.dot(x.T,error) + (self.lambda_/m)*w # nx1
        return partial
    
    def update_weights(self,x,w,y):
        """
        Update the model weights
        INPUT: data x, targets y, current weights w
        OUTPUT: updated weights
        """
        partial = self.grad(x,w,y)
        w = w - self.alpha*partial
        return w
    
    def get_mini_batch(self,x,y,i):
        """
        Get a minibatch of the data
        INPUT: data x, targets y, iteration i
        OUTPUT: subset of the data X,y
        """
        x_mini = x[i*self.batchSize:(i+1)*self.batchSize,:]
        y_mini = y[i*self.batchSize:(i+1)*self.batchSize,:]
        return x_mini,y_mini
    
    def add_bias(self,x):
        """
        Add a column of 1's as the first column in the data x
        INPUT: data x
        OUTPUT: data x with a column of 1's as first column
        """
        x_bias = np.ones((x.shape[0],1))
        x = np.concatenate((x_bias,x),axis=1)
        return x
    
    def init_weights(self,x):
        """
        Initialize the model weights at random
        INPUT: data x
        OUTPUT: random weights
        """
        return (np.random.random((x.shape[1],1))*2 - 1)*1e-2
    
    def fit(self,x,y):
        """
        Fit a model to the data using mini batch gradient descent
        INPUT: data x, targets y
        OUTPUT: model weights w
        """
        if np.all(x[:,0] == np.ones(x.shape[0])):
            pass
        else:
            x = self.add_bias(x)
            
        w = self.init_weights(x)
        n = np.arange(len(x))
        
        """
        Perform mini batch gradient descent
        """
        J = []
        for epoch in range(1, self.n_iter + 1):
            for i in range(0,round(x.shape[0]/self.batchSize)):
                X_mini,Y_mini = self.get_mini_batch(x=x,y=y,i=i)
                J.append(self.cost(x=X_mini,y=Y_mini,w=w))
                w = self.update_weights(x=x,w=w,y=y)

        return w
    
    def predict(self,x,w):
        """
        Predict the target of new input data
        INPUT: data x, learned weights w
        OUTPUT: predicted targets
        """
        x_bias = np.ones((x.shape[0],1))
        x = np.concatenate((x_bias,x),axis=1)
        return np.dot(x,w)
# The features we will be using to predict price
[col for col in list(df.columns) if col not in ['id','price']]
# Create matrices X and Y for training and testing
X = df.as_matrix(columns=[col for col in list(df.columns) if col not in ['id','price']])
Y = df.as_matrix(columns=['price'])
X_test = df_test.as_matrix(columns=[col for col in df_test if col not in ['id','price']])
Y_test = df_test.as_matrix(columns=['price'])
def CrossValidate(x,y,alpha,lambda_):
    """
    Use cross validation on the development data set to obtain the optimal hyperparameters
    INPUT: data x, targets y, alpha (start, end, num), lambda (start, end, num)
    OUTPUT: errs_dict, a dictionary of (alpha,lambda) keys and SSE error values
    """
    
    alphas = np.linspace(alpha[0],alpha[1],alpha[2])
    lambdas_ = np.logspace(lambda_[0],lambda_[1],lambda_[2]).astype(int)
    
    # Use k-fold (10-fold) cross val
    k=10
    cv = KFold(n_splits=k, shuffle=False)
    train_indices = [i[0] for i in cv.split(x)]
    val_indices = [i[1] for i in cv.split(x)]
    
    errs_dict = {(a,l):0 for a,l in zip(alphas,lambdas_)}

    # Loop over alphas and lambdas hyperparams
    for a in alphas:
        print("alpha:",a)
        for lam in lambdas_:
            cv_sse_err = []
            for i in range(k):
                
                x_train = x[train_indices[i],:]
                y_train = y[train_indices[i],:]
                
                x_val = x[val_indices[i],:]
                y_val = y[val_indices[i],:]
                
                model = RidgeGD(lambda_=lam, alpha=a, batchSize=64, n_iter=50, output_every=50)
                W = model.fit(x=x_train,y=y_train)
                
                sse = sum((model.predict(x_val,W) - y_val)**2)[0]
                cv_sse_err.append(sse)
                
            errs_dict[(a,lam)] = (1/k)*sum(cv_sse_err)
    
    return errs_dict
# This takes about 10 minutes to run
# errs_dict = CrossValidate(X,Y,(0.5,0.69,5),(1,2,5))
# Use the minimum value to train the final model
# min(errs_dict, key=errs_dict.get)
# Train model with optimim hyperparams above
model = RidgeGD(lambda_=32,alpha=0.64, batchSize=32, n_iter=200, output_every=50)
# Model weights
W = model.fit(x=X,y=Y)
class Metrics:
    """
    Metrics we will use in evaluating our models
    """
    
    def r_squared(self, model, x, y, w=None, scikit_flag=False):
        """
        Calculate the R^2 value of an input model
        INPUT: model, X (data), y (targets), W (weights, if applicable), scikit_flag (if using scikit)
        """
        if not scikit_flag:
            f = model.predict(x=x, w=w)
            sstot = ((y - np.mean(y))**2).sum()
            ssres = ((y - f)**2).sum()
            return 1 - (ssres/sstot)
        else:
            return model.score(X=x, y=y)
    
    def rss(self, model, x, y, w=None, scikit_flag=False):
        if not scikit_flag:
            f = model.predict(w=w, x=x)
            return ((y - f)**2).sum()
        else:
            f = model.predict(X=x)
            return ((y - f)**2).sum()
# train the scikit model
clf = Ridge(alpha=32, solver='saga')
model_sk = clf.fit(X=X, y=Y)
mets = Metrics()
train_r2 = mets.r_squared(model, X, Y, W)
test_r2 = mets.r_squared(model, X_test, Y_test, W)
train_r2_scikit = mets.r_squared(model_sk, X, Y, scikit_flag=True)
test_r2_scikit = mets.r_squared(model_sk, X_test, Y_test, scikit_flag=True)

train_rss = mets.rss(model, X, Y, W)
test_rss = mets.rss(model, X_test, Y_test, W)
train_rss_scikit = mets.rss(model_sk, X, Y, scikit_flag=True)
test_rss_scikit = mets.rss(model_sk, X_test, Y_test, scikit_flag=True)
print("Model\t\t|\tR2 Train\t|\tR2 Test\t\t|\tRSS Train\t|\tRSS Test")
print("---------------------------------------------------------------------------------------------------------------")
print("From Scratch SGD\t", train_r2, "\t", test_r2, "\t", train_rss, "\t", test_rss)
print("Scikit\t\t\t", train_r2_scikit, "\t", test_r2_scikit, "\t", train_rss_scikit, "\t", test_rss_scikit)
# Create simple dataset
# First, cross validate the model to find optimum hyperparms
X_simple = df.as_matrix(columns=['sqft_living'])
# errs_dict = CrossValidate(X_simple,Y,(0.98,1.3,5),(1,2,5))
# Use these optimum hyperparams in simple model
# min(errs_dict, key=errs_dict.get)
Y = df.as_matrix(columns=['price'])
model_simple = RidgeGD(lambda_=10, alpha=1, batchSize=32, n_iter=200, output_every=50)
W_simple = model_simple.fit(x=X_simple,y=Y)
# Make a test data set using only sqft_living
X_test_simple = df_test.as_matrix(columns=['sqft_living'])
clf = Ridge(alpha=100, solver='saga')
model_sk = clf.fit(X=X_simple, y=Y)
mets = Metrics()
train_r2_simple = mets.r_squared(model, X_simple, Y, W_simple)
test_r2_simple = mets.r_squared(model, X_test_simple, Y_test, W_simple)
train_r2_scikit_simple = mets.r_squared(model_sk, X_simple, Y, scikit_flag=True)
test_r2_scikit_simple = mets.r_squared(model_sk, X_test_simple, Y_test, scikit_flag=True)

train_rss_simple = mets.rss(model, X_simple, Y, W_simple)
test_rss_simple = mets.rss(model, X_test_simple, Y_test, W_simple)
train_rss_scikit_simple = mets.rss(model_sk, X_simple, Y, scikit_flag=True)
test_rss_scikit_simple = mets.rss(model_sk, X_test_simple, Y_test, scikit_flag=True)
print("Model\t\t|\tR2 Train\t|\tR2 Test\t\t|\tRSS Train\t|\tRSS Test")
print("---------------------------------------------------------------------------------------------------------------")
print("From Scratch SGD\t", train_r2_simple, "\t", test_r2_simple, "\t", train_rss_simple, "\t", test_rss_simple)
print("Scikit\t\t\t", train_r2_scikit_simple, "\t\t", test_r2_scikit_simple, "\t", train_rss_scikit_simple, "\t", test_rss_scikit_simple)
def line(m,b,x):
    """
    Return the points on a line
    INPUT: m (line slope), b (line intersect / bias), x (x values of the line)
    OUTPUT: an array of y values for every corresponding x value
    """
    return m*x+b

# Plot appropriate weights and biases and data to see how well line fits through data
l = line(m=W_simple[1],b=W_simple[0],x=np.linspace(-2,12,1000))
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X_simple,Y,s=10,alpha=0.5)
ax.plot(np.linspace(-2,12,1000),l,color='r')
ax.set_ylabel("Price")
ax.set_xlabel("Scaled Sqft House Area")
plt.show()
print("Model\t\t\t|\tR2 Train\t|R2 Test\t|RSS Train\t|\tRSS Test")
print("---------------------------------------------------------------------------------------------------------------")
print("From Scratch SGD (complex)\t", round(train_r2,2), "\t\t", round(test_r2,2), "\t\t", round(train_rss,2), "\t", round(test_rss,2))
print("Scikit (complex)\t\t", round(train_r2_scikit,2), "\t\t", round(test_r2_scikit,2), "\t\t", round(train_rss_scikit,2), "\t", round(test_rss_scikit,2))
print("From Scratch SGD (simple)\t", round(train_r2_simple,2), "\t\t", round(test_r2_simple,2), "\t\t", round(train_rss_simple,2), "\t", round(test_rss_simple,2))
print("Scikit (simple)\t\t\t", round(train_r2_scikit_simple,2), "\t\t", round(test_r2_scikit_simple,2), "\t\t", round(train_rss_scikit_simple,2), "\t", round(test_rss_scikit_simple,2))