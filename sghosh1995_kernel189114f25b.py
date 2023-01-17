import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

###########  Importing Dataset    #######################
dataset = pd.read_csv('../input/my-computer-data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

##########   Encoding Categorial Data  ###################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct  = ColumnTransformer(transformers =[('encoder',OneHotEncoder(),
        [3])], remainder ='passthrough')
X = np.array(ct.fit_transform(X))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

########## Splitting Data into Train and Test data ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 0)
#########  Only features scaling   ######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:] = sc.fit_transform(X_train[:, 0:])
X_test[:, 0:] = sc.transform(X_test[:, 0:])

train_ones = np.ones((len(X_train), 1))
test_ones = np.ones((len(X_test), 1))
X_train = np.hstack((train_ones, X_train))
X_test = np.hstack((test_ones, X_test))
def cost_fun(X, y, theta):
    return 1/(2*len(X))*(sum((np.dot(X, theta)-y)**2))
def grad_fun(X, y, theta):
    return (1/len(X))*np.dot(X.T,(np.dot(X, theta)-y))
def theta_update(X, y, theta, alpha):
    return theta - alpha*grad_fun(X, y, theta)
theta = np.zeros(X_train.shape[1])
Cost_List = []
alpha = 3*1e-3
Iter_List = []
for j in range(1000):
    temp = cost_fun(X_train, y_train, theta)
    if(temp <= 0):
        break
    else:
        Cost_List.append(temp)
        Iter_List.append(j)
        theta = theta_update(X_train, y_train, theta, alpha)
Cost = np.array(Cost_List)
Iter = np.array(Iter_List)
plt.plot(Iter, Cost)
plt.title('Cost vs No of iterations')
plt.xlabel('No of iterations')
plt.ylabel('Cost')
plt.show()
error = (1/(2*len(X_test)))*sum((np.dot(X_test, theta)-y_test)**2)
