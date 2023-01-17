# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualization

from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load Data

train_raw = pd.read_csv("../input/train.csv")

test_raw = pd.read_csv("../input/test.csv")

gs_raw = pd.read_csv("../input/gender_submission.csv")
# Data Insights - Training

train_raw.describe(include="all")

print(np.nan_to_num(train_raw['Cabin']))
train_raw.dtypes


print(train_raw['Pclass'].T)

print(train_raw['Cabin'].T)
# Data Insigts (Testing)

test_raw.describe(include="all")
# Data Insigts (Gender Submission)

gs_raw.describe(include="all")

# Mapping function Extracting PassengerID, Survival, Feature vector phi_prime 

def get_features(X):

    # PassengerId

    PassID = X['PassengerId']

    m = PassID.shape[0]

    nx = 20 # Number of features, this is hardcoded for the timebeing

    # Add two more columns for embarked, and one more for sex

    # Add 7 more for cabin

    # Features 

    X_prime = np.zeros((m,nx))

    # 0 Pclass

    X_prime[:,0] = X['Pclass'] - 2

    # 1 Name           

    # 2,3 Sex

    X_prime[:,2] = 1* (X['Sex'] == 'female')

    X_prime[:,3] = 1* (X['Sex'] == 'male')

    

    # 4 Age           

    mu = 30

    sigma = 14 

    X_prime[:,4] = (X['Age'] - mu ) / sigma

    # 5 SibSp          

    X_prime[:,5] = X['SibSp'] - 3

    # 6 Parch     

    X_prime[:,6] = X['Parch'] - 2

    # 7 Ticket         

    # 8 Fare    

    mu = 33

    sigma = 52 

    X_prime[:,8] = (X['Fare'] - mu ) / sigma

    # 9 Cabin   

    X_prime[:,9] = X['Cabin'].str.contains('A')*1

    X_prime[:,10] = X['Cabin'].str.contains('B')*1

    X_prime[:,11] = X['Cabin'].str.contains('C')*1

    X_prime[:,12] = X['Cabin'].str.contains('D')*1

    X_prime[:,13] = X['Cabin'].str.contains('E')*1

    X_prime[:,14] = X['Cabin'].str.contains('F')*1

    X_prime[:,15] = X['Cabin'].str.contains('G')*1

    X_prime[:,16] = X['Cabin'].str.contains('T')*1

    # Extend Cabin to pick all possible letters, and the numeric value 

    # 17,18,19 Embarked  {C,Q,S}

    selector = X['Embarked'] == 'C'

    X_prime[:,17] = selector * 1 



    selector = X['Embarked'] == 'Q'

    X_prime[:,18] = selector * 1 



    selector = X['Embarked'] == 'S'

    X_prime[:,19] = selector *1

    return PassID, np.nan_to_num(X_prime)
def kernel(X):

    K_x = np.concatenate((X, np.exp(-X**2)), axis=1)

    return K_x
# Neuron 

class Neuron:

    def __init__(self, feature_size, alpha, act=2):

        self.w = np.random.random_sample((feature_size,1))*0.1

        self.b = 0.0

        self.alpha = alpha

        self.activation_function = act

    def activate(self, z):

        # based on set activation function set make the calculation

        if self.activation_function == 2: # Sigmoid

            return 1 / (1 + np.exp(-z))

        if self.activation_function == 3: # tanh

            return 2 / (1 + np.exp(-z)) - 1

        if self.activation_function == 7: # ReLU

            return (z>0)*z

        if self.activation_function == 21: # Gaussian

            return np.exp(-z**2)

    def gradient(self, y_hat, err):

        

        m = y_hat.shape[0] # Sample size

        #da = y/y_hat + (1-y)/(1-y_hat)

        # based on set activation function set make the calculation

        if self.activation_function == 2: # Sigmoid

            return  err*y_hat*(1-y_hat)

        if self.activation_function == 3: # tanh

            return err*(1-y_hat**2)

        if self.activation_function == 7: # ReLU

            return (y_hat>0)*err

        if self.activation_function == 21: # Gaussian

            return -2*y_hat*np.exp(-y_hat**2)*err



    def forward(self, X):

        # given feature vector X, find activation response

        z = np.dot(X,self.w)+self.b

        a = self.activate(z)

        return a

    def backward(self, X, err):

        m = y.shape[0]

        y_hat = self.forward(X)

        grad = self.gradient(y_hat,err)

        self.b -= self.alpha * np.sum(grad)/m

        self.w -= self.alpha * np.dot(X.T,grad)/m

    def print_params(self):

        print("b = {}, w = {}".format(self.b,self.omega.T))
# TODO: Cross Validation Generator, Expects input features and labels, and returns equally sized samples 

def cross_validated(X, n_samples):

    kf = KFold(n_samples, shuffle = True)

    result = [group for group in kf.split(X)]

    

    return result
def ssd(y_hat, y):

    m = y.shape[0]

    return np.dot((y_hat-y).T,y_hat-y)/m


PassID, X = get_features(train_raw)

X = X

y = np.array(train_raw['Survived'])

y = np.reshape(y,(-1,1))

iterations = 3000

instances = 5



folds = 3

oinst = folds

cv_groups = cross_validated(X, folds)



n_x = X.shape[1] # Features

alpha = 0.05

alph = np.ones(oinst)*alpha + (np.random.random_sample(oinst)-0.5)*0.0001

# Setup single neurons

N_g = [Neuron(n_x,alpha,21) for j in range(oinst)]

final_cost = []



for j in range(oinst):

    cost_g = []

    N_g[j].alpha = alph[j]

    

    # Prepare Training and testing sets 

    

    X_train = X[cv_groups[j][1],:] 

    y_train = np.reshape(y[cv_groups[j][1],0],(-1,1)) 

    X_test = X[cv_groups[j][0],:] 

    y_test = np.reshape(y[cv_groups[j][0],0],(-1,1))

    

    for i in range(iterations):

        # Evaluate

        a_g = N_g[j].forward(X_train)

        

        # Learn

        N_g[j].backward(X_train,a_g - y_train)

        

        # Evaluate

        # a_g = N_g[j].forward(X_test)

        

        # Performance vote

        # cost_g.append(np.sum(np.abs(a_g-y_test))/y.shape[0])

    

    a_g = (N_g[j].forward(X_test)>0.5)*1

    m = y_test.shape[0]

    

    final_cost.append(1-ssd(a_g,y_test))



    print("Testing:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y_test)))



    #plt.plot(cost_g)

    #plt.title("Cost over time")

    #plt.show()



    #plt.scatter(a_g, y)

    #plt.show()

final_cost = np.reshape((final_cost),(-1,1)).tolist()     



plt.plot(final_cost)    

plt.show()



PassID, X = get_features(test_raw)

X2 = X



best_of_breed = final_cost.index(max(final_cost))

a_gm = N_g[best_of_breed].forward(X2)

    

y_hat = np.reshape(a_gm,(-1,1))



print(y_hat.shape)

data = y_hat > 0.5

data = data*1

s0 = pd.Series(PassID, index=PassID)

s1 = pd.Series(data[:,0], index=PassID)



df = pd.DataFrame(data = s1,index = PassID)

df.columns = ['Survived']

df.to_csv('best_of_breed_gauss_nn_3.csv', sep=',')


PassID, X = get_features(train_raw)

X = X

y = np.array(train_raw['Survived'])

y = np.reshape(y,(-1,1))

iterations = 5000

instances = 5



folds = 3

oinst = folds

cv_groups = cross_validated(X, folds)



n_x = X.shape[1] # Features

alpha = 0.25

alph = np.ones(oinst)*alpha + (np.random.random_sample(oinst)-0.5)*0.0001

# Setup Hidden Layer

N_21 = [Neuron(n_x,alpha,3) for j in range(oinst)]

N_22 = [Neuron(n_x,alpha,3) for j in range(oinst)]

# Setup output layer

N_g = [Neuron(2,alpha,2) for j in range(oinst)]

final_gain = []

all_gain = []

for j in range(oinst):

    cost_g = []

    N_g[j].alpha = alph[j]

    N_21[j].alpha = alph[j]

    N_22[j].alpha = alph[j]

    

    # Prepare Training and testing sets 

    X_train = X[cv_groups[j][1],:] 

    y_train = np.reshape(y[cv_groups[j][1],0],(-1,1)) 

    X_test = X[cv_groups[j][0],:] 

    y_test = np.reshape(y[cv_groups[j][0],0],(-1,1))

    

    for i in range(iterations):

        # Evaluate

        a_21 = N_21[j].forward(X_train) 

        a_22 = N_22[j].forward(X_train)  

        a_2 = np.concatenate((a_21,a_22),axis=1)

        a_g = N_g[j].forward(a_2)

        

        # Learn

        grad = N_g[j].gradient(a_g,a_g - y_train)

        N_g[j].backward(a_2,a_g - y_train)

        

        N_21[j].backward(X_train,grad*N_g[j].w[0])

        N_22[j].backward(X_train,grad*N_g[j].w[1])

    

    

    # Finally Evaluate Model 

    a_21 = N_21[j].forward(X_test) 

    a_22 = N_22[j].forward(X_test)  

    a_2 = np.concatenate((a_21,a_22),axis=1)

    a_g = N_g[j].forward(a_2)

    #plt.scatter(a_g, y_test)

    #plt.show()

    a_g = (a_g>0.5)*1

    m = y_test.shape[0]

    final_gain.append(1-ssd(a_g,y_test))



    print("Testing:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y_test)))



    # Evaluate All

    a_21 = N_21[j].forward(X) 

    a_22 = N_22[j].forward(X)  

    a_2 = np.concatenate((a_21,a_22),axis=1)

    a_g = N_g[j].forward(a_2)

    #plt.scatter(a_g, y)

    #plt.show()

    a_g = (a_g>0.5)*1

    m = y.shape[0]

    all_gain.append(1-ssd(a_g,y))

    print("Testing All:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y)))

    #plt.plot(cost_g)

    #plt.title("Cost over time")

    #plt.show()





final_gain= np.reshape((final_gain),(-1,1)).tolist()         

all_gain= np.reshape((all_gain),(-1,1)).tolist()   

plt.plot(final_gain)

plt.plot(all_gain)

plt.show()

PassID, X = get_features(test_raw)

X2 = X



best_of_breed = final_gain.index(max(final_gain))

a_21 = N_21[best_of_breed].forward(X2) 

a_22 = N_22[best_of_breed].forward(X2)  

a_2 = np.concatenate((a_21,a_22),axis=1)

a_gm = N_g[best_of_breed].forward(a_2)

    

y_hat = np.reshape(a_gm,(-1,1))



print(y_hat.shape)

data = y_hat > 0.5

data = data*1

s0 = pd.Series(PassID, index=PassID)

s1 = pd.Series(data[:,0], index=PassID)



df = pd.DataFrame(data = s1,index = PassID)

df.columns = ['Survived']

df.to_csv('best_of_2tanh_1sig.csv', sep=',')


PassID, X = get_features(train_raw)

X = X

y = np.array(train_raw['Survived'])

y = np.reshape(y,(-1,1))

iterations = 5000

instances = 4

alph = np.linspace(0.1, 0.40, instances)



folds = 3

oinst = folds

cv_groups = cross_validated(X, folds)



n_x = X.shape[1] # Features

final_gain = []

all_gain = []

alphas = []

for a in range(instances):

    alpha = alph[a]+(np.random.random_sample()-0.5)*0.0001

    # Setup Hidden Layer

    N_21 = [Neuron(n_x,alpha,3) for j in range(oinst)]

    N_22 = [Neuron(n_x,alpha,3) for j in range(oinst)]

    # Setup output layer

    N_g = [Neuron(2,alpha,2) for j in range(oinst)]



    for j in range(oinst):

        alpha = alph[a]+(np.random.random_sample()-0.5)*0.00001

        N_g[j].alpha = alpha

        N_21[j].alpha = alpha

        N_22[j].alpha = alpha

        alphas.append(alpha)

        # Prepare Training and testing sets 

        X_train = X[cv_groups[j][1],:] 

        y_train = np.reshape(y[cv_groups[j][1],0],(-1,1)) 

        X_test = X[cv_groups[j][0],:] 

        y_test = np.reshape(y[cv_groups[j][0],0],(-1,1))

    

        for i in range(iterations):

            # Evaluate

            a_21 = N_21[j].forward(X_train) 

            a_22 = N_22[j].forward(X_train)  

            a_2 = np.concatenate((a_21,a_22),axis=1)

            a_g = N_g[j].forward(a_2)

        

            # Learn

            grad = N_g[j].gradient(a_g,a_g - y_train)

            N_g[j].backward(a_2,a_g - y_train)

        

            N_21[j].backward(X_train,grad*N_g[j].w[0])

            N_22[j].backward(X_train,grad*N_g[j].w[1])

    

    

        # Finally Evaluate Model 

        a_21 = N_21[j].forward(X_test) 

        a_22 = N_22[j].forward(X_test)  

        a_2 = np.concatenate((a_21,a_22),axis=1)

        a_g = N_g[j].forward(a_2)

        print("Testing Sample:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y_test)))

        #plt.scatter(a_g, y_test)

        #plt.show()

        a_g = (a_g>0.5)*1

        m = y_test.shape[0]

        final_gain.append(1-ssd(a_g,y_test))





        # Evaluate All

        a_21 = N_21[j].forward(X) 

        a_22 = N_22[j].forward(X)  

        a_2 = np.concatenate((a_21,a_22),axis=1)

        a_g = N_g[j].forward(a_2)

        #plt.scatter(a_g, y)

        #plt.show()

        a_g = (a_g>0.5)*1

        m = y.shape[0]

        all_gain.append(1-ssd(a_g,y))

        print("Testing All:[alpha ={}]  Success = {}".format(N_g[j].alpha,1-ssd(a_g,y)))

        #plt.plot(cost_g)

        #plt.title("Cost over time")

        #plt.show()





final_gain= np.reshape((final_gain),(-1,1)).tolist()         

all_gain= np.reshape((all_gain),(-1,1)).tolist()   

plt.scatter(alphas,final_gain)

plt.scatter(alphas,all_gain)

plt.show()

 