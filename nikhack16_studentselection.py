# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from scipy.optimize import minimize



from sklearn.preprocessing import PolynomialFeatures



pd.set_option('display.notebook_repr_html', False)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 150)

pd.set_option('display.max_seq_items', None)



%matplotlib inline



import seaborn as sns

sns.set_context('notebook')

sns.set_style('white')





def loadData(file,delimiter):

    data = np.loadtxt(file , delimiter = delimiter)

    print('Dimensions :' ,data.shape)

    print(data[0:6,:])

    return data



def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):

    neg = data[:,2] == 0

    pos = data[:,2] == 1

    

    if axes == None:

        axes = plt.gca()

    axes.scatter(data[pos][:,0] , data[pos][:,1] , marker = '+' , c = 'k' , s = 60 , linewidth = 2, label = label_pos)

    axes.scatter(data[neg][:,0] , data[neg][:,1] ,c = 'y', s = 60 , label = label_neg)

    axes.set_xlabel(label_x)

    axes.set_ylabel(label_y)

    axes.legend(frameon = True , fancybox = True)

    

    

## Logistic Regression



data = loadData('../input/ex2data1.txt' , ',')

#print(data)

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]

y = np.c_[data[:,2]]





#plotData(data,'Exam 1 Score','Exam 2 Score','Admitted','Not Admitted')



def sigmoid(z):

    return (1 / (1 + np.exp(-z)))



def costFunc(theta,X,y):

    m = y.size

    h = sigmoid(X.dot(theta))

    

    J = -(1/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))

    

    if np.isnan(J[0]):

        return np.inf

    return (J[0])



def gradient(theta,X,y):

    m = y.size

    h = sigmoid(X.dot(theta.reshape(-1,1)))

    

    grad = (1/m)*X.T.dot(h-y)

    

    return (grad.flatten())



initial_theta = np.zeros(X.shape[1])

cost = costFunc(initial_theta,X,y)

grad = gradient(initial_theta,X,y)

print('Cost: \n', cost)

print('Grad: \n', grad)



    

res = minimize(costFunc,initial_theta,args=(X,y), method = None , jac = gradient, options={'maxiter':400})

res

    

    

def predict(theta,X,threshold = 0.5):

    p = sigmoid(X.dot(theta.T)) >= threshold

    return (p.astype('int'))



pr = sigmoid(np.array([1, 50, 75]).dot(res.x.T))

print(pr)

p = predict(res.x,X)

if p.size!=0:

    print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

    

    

# Decision Boundary



plt.scatter(50, 75, s=60, c='r', marker='v', label='(50, 75)')

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')

x1_min, x1_max = X[:,1].min(), X[:,1].max(),

x2_min, x2_max = X[:,2].min(), X[:,2].max(),

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))

h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');

    

    





    

    






