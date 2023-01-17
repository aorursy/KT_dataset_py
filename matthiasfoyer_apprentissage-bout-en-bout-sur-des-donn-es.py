# Importation du dataset

import pandas as pd
gaussian_df = pd.read_csv('../input/gaussian_data.csv')
# Split: 1 set to fit, 1 set to test/analyse

test_final_df = gaussian_df.sample(frac=0.2, random_state=42)
gaussian_df = gaussian_df.drop(test_final_df.index)
# One-hot encoding

X = gaussian_df.iloc[:,0:2]
y = pd.get_dummies(gaussian_df.iloc[:,2])
# Split: 1 set to train, 1 set to test (fitting)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# Importation utility.py

import numpy as np
from scipy.special import softmax # use built-in function to avoid numerical instability

class Utility:
    @staticmethod
    def identity(Z):
        return Z,1
    
    @staticmethod
    def tanh(Z):
        """
        Z : non activated outputs
        Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
        """
        A = np.empty(Z.shape)
        A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
        df = 1-A**2
        return A,df
    
    @staticmethod
    def sigmoid(Z):
        A = np.empty(Z.shape)
        A = 1.0 / (1 + np.exp(-Z))
        df = A * (1 - A)
        return A,df
    
    @staticmethod
    def relu(Z):
        A = np.empty(Z.shape)
        A = np.maximum(0,Z)
        df = (Z > 0).astype(int)
        return A,df
    
    @staticmethod
    def softmax(Z):
        return softmax(Z, axis=0) # from scipy.special
    
    @staticmethod
    def cross_entropy_cost(y_hat, y):
        n  = y_hat.shape[1]
        ce = -np.sum(y*np.log(y_hat+1e-9))/n
        return ce
    
    """
    Explication graphique du MSE:
    https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f
    """
    @staticmethod
    def MSE_cost(y_hat, y):
        mse = np.square(np.subtract(y_hat, y)).mean()
        return mse
# Take a dataframe instance, return a numpy array

def toList(df,i):
    return np.transpose(df.iloc[[i]].values)
# Neural network

import statistics
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, X_train, y_train, X_test, y_test, hidden_layers_sizes, activation, learning_rate, epoch):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.hidden_layers_sizes = hidden_layers_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.n_layers = len(hidden_layers_sizes)
        
        # Initializaing weights matrices
        self.weights = [None] * (self.n_layers + 1)
        for i in range(self.n_layers + 1):
            if i == 0:
                self.weights[i] = self.__weights_initialization(len(self.X_train.columns), self.hidden_layers_sizes[i])
            elif i == (self.n_layers):
                self.weights[i] = self.__weights_initialization(self.hidden_layers_sizes[i-1], len(self.y_train.columns))
            else:
                self.weights[i] = self.__weights_initialization(self.hidden_layers_sizes[i-1], self.hidden_layers_sizes[i])
                
        # Initializing bias matrices
        self.bias = [None] * (self.n_layers + 1)
        for i in range(self.n_layers + 1):
            if i == (self.n_layers):
                self.bias[i] = self.__bias_initialization(len(self.y_train.columns))
            else:
                self.bias[i] = self.__bias_initialization(self.hidden_layers_sizes[i])
        
        # Initializing the derivative list
        self.df = [None]*(self.n_layers + 1)
        
        # Initializing the application list
        self.A = [None]*(self.n_layers + 1)
        
    def __weights_initialization(self, X, y):
        return np.random.uniform(low=-1.0, high=1.0, size=(y*X)).reshape(y,X)
    
    def __bias_initialization(self, y):
        return np.random.uniform(low=0.0, high=0.0, size=y).reshape(y,1)
    
    # Forward propagation
    def forwardPropagation(self, X, y, i):
        for k in range(self.n_layers + 1):
            if k == 0:
                A = np.transpose(X.iloc[[i]].values)
                W = self.weights[k]
                b = self.bias[k]
                Z = np.add(np.matmul(W,A), b)
                self.A[k], self.df[k] = self.activation(Z)
            elif k == self.n_layers:
                A = self.A[k-1]
                W = self.weights[k]
                b = self.bias[k]
                Z = np.add(np.matmul(W,A), b)
                self.A[k] = Utility.softmax(Z)
            else:
                A = self.A[k-1]
                W = self.weights[k]
                b = self.bias[k]
                Z = np.add(np.matmul(W,A), b)
                self.A[k], self.df[k] = self.activation(Z)

        # Error
        y_hat = self.A[-1]
        y = toList(y, i)
        return Utility.cross_entropy_cost(y_hat, y)
    
    # X le set, i la position de l'instance dans le set
    def predict(self, X, i):
        for k in range(self.n_layers + 1):
            if k == 0:
                A = np.transpose((X.iloc[[i]].values))
                W = self.weights[k]
                b = self.bias[k]
                Z = np.add(np.matmul(W,A), b)
                self.A[k], self.df[k] = self.activation(Z)
            elif k == self.n_layers:
                A = self.A[k-1]
                W = self.weights[k]
                b = self.bias[k]
                Z = np.add(np.matmul(W,A), b)
                self.A[k] = Utility.softmax(Z)
            else:
                A = self.A[k-1]
                W = self.weights[k]
                b = self.bias[k]
                Z = np.add(np.matmul(W,A), b)
                self.A[k], self.df[k] = self.activation(Z)
        
        # Prediction
        return self.A[-1]
    
    # Backward propagation
    def backwardPropagation(self, X, y):
        delta = [None] * (self.n_layers + 1)
        dW = [None] * (self.n_layers + 1)
        db = [None] * (self.n_layers + 1)
        
        delta[-1] = (self.A[-1] - y)
        dW[-1] = np.matmul(delta[-1], np.transpose(self.A[-2]))
        db[-1] = delta[-1]
        
        for i in range(self.n_layers):
            l = self.n_layers - 1 - i
            delta[l] = np.multiply(np.matmul(np.transpose(self.weights[l+1]), delta[l+1]), self.df[l])
            if l == 0:
                dW[l] = np.matmul(delta[l], np.transpose(X))
            else:
                dW[l] = np.matmul(delta[l], np.transpose(self.A[l-1]))
            db[l] = delta[l]
    
        for j in range(self.n_layers + 1):
            self.weights[j] = np.subtract(self.weights[j], np.dot(self.learning_rate, dW[j]))
            self.bias[j] = np.subtract(self.bias[j], np.dot(self.learning_rate, db[j]))
    
    # Model training
    def fit(self):
        erreur_train_list = ([],[])
        erreur_test_list = ([],[])
        for e in range(self.epoch):
            # Shuffle dataset
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
            
            # Training
            erreur_train = []
            for i in range(len((self.X_train).index)):
                erreur_train.append(self.forwardPropagation(self.X_train, self.y_train, i))
                self.backwardPropagation(toList(self.X_train, i), toList(self.y_train, i))
            erreur_train_list[0].append(e)
            erreur_train_list[1].append(statistics.mean(erreur_train))
            
            # Testing
            erreur_test = []
            for i in range(len((self.X_test).index)):
                erreur_test.append(self.forwardPropagation(self.X_test, self.y_test, i))
            erreur_test_list[0].append(e)
            erreur_test_list[1].append(statistics.mean(erreur_test))
        
        # Plot results
        plt.plot(erreur_train_list[0], erreur_train_list[1], label='Train')
        plt.plot(erreur_test_list[0], erreur_test_list[1], label='Test')
        plt.xlabel('Epoch of training')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
# Model fitting

model = NeuralNet(X_train, y_train, X_test, y_test, (4,3,2), Utility.tanh, 0.01, 200)
model.fit()
# Predicted results

y_pred = np.empty((0,3))
for i in range (len(X_test)):
    y_pred = np.vstack((y_pred, model.predict(X_test, i).transpose()))
np.set_printoptions(precision=3, suppress=True)

# Show 5 first lines
print(y_pred[:6])
# Actual results

y_actual = y_test.to_numpy()

# Show 5 first lines
print(y_actual[:6])
print('La classe prédite pour la donnée 0 est la classe', np.argmax(y_pred[0]))
# Ratio

def ratio (y_pred, y_actual):
    avg = 0
    for i in range (len(y_pred)):
        comp = np.multiply((np.argmax(y_pred[i]) == np.argmax(y_actual[i])), 1)
        avg += (comp - avg) / (i + 1)
    return avg

print("La précision de notre réseau neuronal est de %.2f%%" % (ratio(y_pred, y_actual) * 100))
# Confusion matrix

def confusionMatrix (nb_class, y_pred, y_actual):
    m = np.zeros((nb_class,nb_class), int)
    for i in range (len(y_pred)):
        m[np.argmax(y_pred[i])][np.argmax(y_actual[i])] += 1
    return m
    
confusion_mtx = confusionMatrix(3, y_pred, y_actual)
print(confusion_mtx)
# Show confusion matrix

import seaborn as sns
class_names = ["class-0", "class-1", "class-2"]
plt.figure(figsize = (8,8))
sns.set(font_scale=2) # label size
ax = sns.heatmap(confusion_mtx, annot=True, annot_kws={"size": 30}, # font size
            cbar=False, cmap="Blues", fmt="d", # format (int)
            xticklabels=class_names, yticklabels=class_names)
ax.set(title="Matrice de Confusion", xlabel="Actual", ylabel="Predicted")
plt.show()