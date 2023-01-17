import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in sorted(filenames):
        print('🟢' if filename.startswith('ex2') else '❌', os.path.join(dirname, filename))
import numpy as np 
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/machine-learning-by-andrew-ng/ex2data1.csv')
df['e1score'].describe()
df['e2score'].describe()
def plot_data(): 
    # positive
    p1 = plt.scatter(
        df[df.admission == 1]['e1score'], 
        df[df.admission == 1]['e2score'], 
        color="r", 
        marker="+",
        s=30
    )
    
    # negative
    p2 = plt.scatter(
        df[df.admission == 0]['e1score'], 
        df[df.admission == 0]['e2score'], 
        color="y", 
        marker="o",
        s=30
    )
    
    # putting labels 
    plt.xlabel('e1 score') 
    plt.ylabel('e2 score')
    plt.legend([p1, p2], ['admitted', 'not admitted'], loc='lower left')
    
    # title
    plt.title('Data Visualization')
  
    # function to show plot 
    plt.show()

plot_data()
X = pd.DataFrame([df.e1score, df.e2score]).transpose().to_numpy()
m, n = X.shape

X = np.append(np.ones([m, 1]), X, axis=1)
X.shape
y = df.admission.to_numpy()
y.size
def sigmoid(z):
    return np.ones(z.shape) / (1 + np.exp(-z))

sigmoid(np.zeros([1,1]))
def sigmoid_cost(theta, X, y):
    m = y.size
    z = X.dot(theta)
    h = sigmoid(z)
    J = np.sum(- y.dot(np.log(h)) - (1-y).dot(np.log(1-h)))
#     J = -y.transpose().dot(np.log(h)) - (1-y.transpose()).dot(np.log(1-h))
    return J / m

sigmoid_cost(
    theta=np.zeros([n+1, 1]),
    X=X,
    y=y
)
def sigmoid_grad(theta, X, y):
    m = y.size
    z = X.dot(theta)    
    h = sigmoid(z)
    grad = 1/m * np.sum(np.dot((h-y), X), axis=0)
#     grad = 1/m * (h-y).transpose().dot(X)
    return grad

sigmoid_grad(
    theta=np.zeros([n+1, 1]),
    X=X,
    y=y
)
options = {'full_output': True, 'maxiter': 400}
theta, cost, _, _, _ = op.fmin(
    lambda t: sigmoid_cost(t, X, y), 
    np.zeros([n+1, 1]),
    **options
)
print()
print('theta:', theta)
print('cost:', cost)
def plot_decision_boundary(): 
    # plotting the actual points as scatter plot    
    p1 = plt.scatter(
        df[df.admission == 1]['e1score'], 
        df[df.admission == 1]['e2score'], 
        color="r", 
        marker="+",
        s=30
    )
    p2 = plt.scatter(
        df[df.admission == 0]['e1score'], 
        df[df.admission == 0]['e2score'], 
        color="y", 
        marker="o",
        s=30
    )
    
    # boundary
    px = np.array([np.min(X[:,1])-2, np.max(X[:,2])+2])
    py = (-1 / theta[2]) * (theta[1]*px + theta[0])
    plt.plot(px, py)
    
    # putting labels 
    plt.xlabel('e1 score') 
    plt.ylabel('e2 score')
    plt.legend([p1, p2], ['admitted', 'not admitted'], loc='lower left')    
    plt.title('Decision Boundary')
  
    # function to show plot 
    plt.show()

plot_decision_boundary()
def predict(e1score, e2score):
    print("For a student with scores %s and %s, we predict an admission probability of %f" % (
        e1score,
        e2score,
        sigmoid(np.array([1, e1score, e2score]).dot(theta))
    ))
predict(45, 85)
predict(30, 45)