import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from ipywidgets import IntProgress

from IPython.display import display
def DataImport(filename):

    """

    Arguments:

        filename -- python str (string) containing the name of the csv file to read the data

    Returns:

        X -- data, numpy array of shape (number of variables, number of examples)

        Y -- true "label" vector (for example: containing 0 if no survived, 1 if survived), shape (1, number of examples)

    """

    Table = pd.read_csv("../input/titanic/"+filename+".csv")

    Table = Table.drop(["Cabin","Name","Ticket"],axis=1)

    Table["Fare"] = Table["Fare"].replace(np.nan,32)/512.

    Table["Age"] = Table["Age"].replace(np.nan,30)/100.

    Table["Embarked"] = Table["Embarked"].replace(np.nan,"C")

    Table["Embarked"] = Table["Embarked"].replace(["C","Q","S"],[0,1,2])/2.

    Table["Sex"] = Table["Sex"].replace(["male","female"],[1,0])

    Table["Pclass"] = 1. - Table["Pclass"]/3.

    Table["SibSp"] = Table["SibSp"]/10.

    if "Survived" in Table.columns:

      X = Table.drop(["PassengerId","Survived"],axis=1).to_numpy().T

      Y = Table["Survived"].to_numpy().reshape((1,-1))

    else:

      X = Table.drop(["PassengerId"],axis=1).to_numpy().T

      Y = Table["PassengerId"].to_numpy().reshape((1,-1))

    return X, Y
X_data, Y_data = DataImport("train")

print("X.shape=",str(X_data.shape))

print("Y.shape=",str(Y_data.shape))

print("X(0).T=",str(X_data[:,0].T))

print("Y(0).T=",str(Y_data[:,0].T))
list_permute = np.random.permutation(X_data.shape[1])

list_train, list_validation = list_permute[:600], list_permute[600:]

X_train, Y_train = X_data[:,list_train], Y_data[:,list_train]

X_validation, Y_validation = X_data[:,list_validation], Y_data[:,list_validation]
# import numpy as np

# import matplotlib.pyplot as plt

# from ipywidgets import IntProgress

# from IPython.display import display

class NeuralNetwork:

    def __init__(self,layer_dims,activations,cost_function):

        """

        Arguments:

        layer_dims   -- python array (list) containing the dimensions of each layer in our network

        activations  -- python array (list) containing the activation function of each layer in our network

        cost_function-- python str containing the cost function of our network

        Returns:

        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                        W -- weight matrix of shape (layer_dims[l], layer_dims[l-1])

                        b -- bias vector of shape (layer_dims[l], 1)

        """

        L = len(layer_dims)

        self.cost_function = cost_function

        self.parameters = {}

        self.activations, self.caches, self.grads = [], [], []

        for l in range(1,L):

            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01

            self.parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

            self.activations.append(activations[l-1])

        

    def activation_forward(self,A_prev, W, b, activation):

        """

        Arguments:

        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

        b -- bias vector, numpy array of shape (size of the current layer, 1)

        activation -- the activation to be used in this layer, stored as a text string: "sigmoid", "relu" or "tanh"

        Returns:

        A -- the output of the activation function, also called the post-activation value 

        cache -- a python tuple containing "linear_cache" and "activation_cache";

                 stored for computing the backward pass efficiently

        """

        Z = np.dot(W,A_prev)+b

        linear_cache = (A_prev, W, b)

        if activation == "sigmoid":

            A = 1/(1+np.exp(-Z));

        elif activation == "relu":

            A = np.maximum(0,Z)

        elif activation == "tanh":

            A = np.tanh(Z)

        activation_cache = Z

        cache = (linear_cache, activation_cache)

        return A, cache

    

    def activation_backward(self,dA, cache, activation):

        """

        Arguments:

        dA -- post-activation gradient for current layer l 

        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently

        activation -- the activation to be used in this layer, stored as a text string: "sigmoid", "relu" or "tanh"

        Returns:

        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

        dW -- Gradient of the cost with respect to W (current layer l), same shape as W

        db -- Gradient of the cost with respect to b (current layer l), same shape as b

        """

        linear_cache, activation_cache = cache

        Z = activation_cache

        if activation == "sigmoid":

            s = 1/(1+np.exp(-Z))

            dZ = dA * s * (1-s)

        elif activation == "relu":

            dZ = np.array(dA, copy=True)

            dZ[Z <= 0] = 0

        elif activation == "tanh":

            s = np.tanh(Z)

            dZ = dA*(1-s**2)

        A_prev, W, b = linear_cache

        m = A_prev.shape[1]

        dW = 1/m*np.dot(dZ,A_prev.T)

        db = 1/m*np.sum(dZ,axis=1,keepdims=True)

        dA_prev = np.dot(W.T,dZ)

        return dA_prev, dW, db

    

    def forward(self,X):

        """

        Arguments:

        X -- data, numpy array of shape (input size, number of examples)

        Returns:

        AL -- last post-activation value

        caches -- list of caches containing: every cache of activation_forward()

        """

        self.caches = []

        A = X

        L = len(self.parameters) // 2

        for l in range(1, L+1):

            A_prev = A

            activation = self.activations[l-1]

            A, cache = self.activation_forward(A_prev, self.parameters["W"+str(l)], self.parameters["b"+str(l)], activation)

            self.caches.append(cache)

        AL = A

        return AL

    

    def compute_cost(self, AL, Y):

        """

        Arguments:

        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)

        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)



        Returns:

        cost -- cross-entropy cost

        """

        m = Y.shape[1]

        if self.cost_function == "cross-entropy":

                cost = (1./m) * (-np.dot(Y,np.log(AL+1e-8).T) - np.dot(1-Y, np.log(1-AL+1e-8).T))

        return cost

    

    def backward(self, AL, Y):

        """

        Arguments:

        AL -- probability vector, output of the forward propagation (L_model_forward())

        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)

        caches -- list of caches containing: every cache of linear_activation_forward() 

        Returns:

        grads -- A dictionary with the gradients

                 grads["dA" + str(l)] = ... 

                 grads["dW" + str(l)] = ...

                 grads["db" + str(l)] = ... 

        """

        grads = {}

        L = len(self.caches)

        m = AL.shape[1]

        Y = Y.reshape(AL.shape)

        if self.cost_function == "cross-entropy":

            dAL = - (np.divide(Y, AL+1e-8) - np.divide(1 - Y, 1 - AL+1e-8))

        grads["dA" + str(L)] = dAL

        for l in reversed(range(L)):

            current_cache, activation = self.caches[l], self.activations[l]

            dA_prev_temp, dW_temp, db_temp = self.activation_backward(grads["dA" + str(l + 1)], current_cache, activation)

            grads["dA" + str(l)] = dA_prev_temp

            grads["dW" + str(l + 1)] = dW_temp

            grads["db" + str(l + 1)] = db_temp

        self.grads = grads

    

    def update_parameters(self, learning_rate):

        """

        Arguments:

        parameters -- python dictionary containing your parameters 

        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:

        parameters -- python dictionary containing your updated parameters 

                      parameters["W" + str(l)] = ... 

                      parameters["b" + str(l)] = ...

        """

        grads = self.grads

        L = len(self.parameters) // 2

        for l in range(L):

            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]

            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        

    def train(self, X, Y, batch_size=None, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

        """

        Arguments:

        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)

        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

        learning_rate -- learning rate of the gradient descent update rule

        num_iterations -- number of iterations of the optimization loop

        print_cost -- if True, it prints the cost every 100 steps



        Returns:

        parameters -- parameters learnt by the model. They can then be used to predict.

        """

        if not batch_size == None:

            m = X.shape[1] // batch_size

            X_batch, Y_batch = [], []

            for ii in range(m-1):

                X_batch.append(X[:,ii*batch_size:(ii+1)*batch_size])

                Y_batch.append(Y[:,ii*batch_size:(ii+1)*batch_size])

            X_batch.append(X[:,(m-1)*batch_size:])

            Y_batch.append(Y[:,(m-1)*batch_size:])

        else:

            X_batch, Y_batch = [X], [Y]

        if print_cost:

          f = IntProgress(min=0, max=num_iterations//1000)

          display(f)

        costs = []

        for i in range(0, int(num_iterations)):

            cost_iteration = []

            for ii in range(len(X_batch)):

                X_mini, Y_mini = X_batch[ii], Y_batch[ii]

                AL= self.forward(X_mini)

                cost = self.compute_cost(AL,Y_mini)

                cost_iteration.append(cost)

                self.backward(AL,Y_mini)

                self.update_parameters(learning_rate)

            if print_cost and i % 1000 == 0:

                costs.append(np.average(cost_iteration))

                f.value += 1

        if print_cost:

            costs.append(np.average(cost_iteration))

            plt.plot(np.squeeze(costs))

            plt.ylabel('cost')

            plt.xlabel('iterations (per thousands)')

            plt.title("Learning rate = " + str(learning_rate))

            plt.show()

        else:

            print("training done:",num_iterations,"iterations")
def Predict(AL):

  return (AL>0.5).astype(int)



def Predict_accuracy(AL,Y):

  Yhat = Predict(AL)

  accuracy = 1 - np.average(np.abs(Yhat-Y))

  return accuracy, Yhat
Logistic = NeuralNetwork((7,1),["sigmoid"],"cross-entropy")



print("Error with training data:",np.squeeze(Logistic.compute_cost( Logistic.forward(X_train), Y_train )))

print("Error with validation data:",np.squeeze(Logistic.compute_cost( Logistic.forward(X_validation), Y_validation )))



acc_t, _ = Predict_accuracy(Logistic.forward(X_train),Y_train)

acc_v, _ = Predict_accuracy(Logistic.forward(X_validation),Y_validation)

print("Accuracy of model with training data:",acc_t*100)

print("Accuracy of model with validation data:",acc_v*100)
Logistic.train(X_train,Y_train,learning_rate=1e-2,num_iterations=10e3,\

               batch_size=200,print_cost=True)
print("Error with training data:",np.squeeze(Logistic.compute_cost( Logistic.forward(X_train), Y_train )))

print("Error with validation data:",np.squeeze(Logistic.compute_cost( Logistic.forward(X_validation), Y_validation )))



acc_t, _ = Predict_accuracy(Logistic.forward(X_train),Y_train)

acc_v, pred_v = Predict_accuracy(Logistic.forward(X_validation),Y_validation)



print("Accuracy of model with training data:",acc_t*100)

print("Accuracy of model with validation data:",acc_v*100)

print(np.concatenate((pred_v.T,Y_validation.T),axis=1)[:10,:])
NN = NeuralNetwork((7,3,1),["relu","sigmoid"],"cross-entropy")



print("Error with training data:",np.squeeze(NN.compute_cost( NN.forward(X_train), Y_train )))

print("Error with validation data:",np.squeeze(NN.compute_cost( NN.forward(X_validation), Y_validation )))



acc_t, pred_t = Predict_accuracy(NN.forward(X_train),Y_train)

acc_v, pred_v = Predict_accuracy(NN.forward(X_validation),Y_validation)

print("Accuracy of model with training data:",acc_t*100)

print("Accuracy of model with validation data:",acc_v*100)
NN.train(X_train,Y_train,learning_rate=1e-2,num_iterations=10e3, \

         batch_size=200,print_cost=True)
print("Error with training data:",np.squeeze(NN.compute_cost( NN.forward(X_train), Y_train )))

print("Error with validation data:",np.squeeze(NN.compute_cost( NN.forward(X_validation), Y_validation )))
acc_t, pred_t = Predict_accuracy(NN.forward(X_train),Y_train)

acc_v, pred_v = Predict_accuracy(NN.forward(X_validation),Y_validation)

print("Accuracy of model with training data:",acc_t*100)

print("Accuracy of model with validation data:",acc_v*100)

print(np.concatenate((pred_v.T,Y_validation.T,),axis=1)[:10,:])
NN_deep = NeuralNetwork((7,6,3,1),["tanh","relu","sigmoid"],"cross-entropy")



print("Error with training data:",np.squeeze(NN_deep.compute_cost( NN_deep.forward(X_train), Y_train )))

print("Error with validation data:",np.squeeze(NN_deep.compute_cost( NN_deep.forward(X_validation), Y_validation )))



acc_t, _ = Predict_accuracy(NN_deep.forward(X_train),Y_train)

acc_v, _ = Predict_accuracy(NN_deep.forward(X_validation),Y_validation)

print("Accuracy of model with training data:",acc_t*100)

print("Accuracy of model with validation data:",acc_v*100)
NN_deep.train(X_train,Y_train,learning_rate=1e-1,num_iterations=10e3,\

               batch_size=200,print_cost=True)
print("Error with training data:",np.squeeze(NN_deep.compute_cost( NN_deep.forward(X_train), Y_train )))

print("Error with validation data:",np.squeeze(NN_deep.compute_cost( NN_deep.forward(X_validation), Y_validation )))



acc_t, pred_t = Predict_accuracy(NN_deep.forward(X_train),Y_train)

acc_v, pred_v = Predict_accuracy(NN_deep.forward(X_validation),Y_validation)

print("Accuracy of model with training data:",acc_t*100)

print("Accuracy of model with validation data:",acc_v*100)

print(np.concatenate((pred_v.T,Y_validation.T,),axis=1)[:10,:])
X_test, Y_test = DataImport("test")

print("X.shape=",str(X_test.shape))

print("X(0).T=",str(X_test[:,0].T))
predict_NN_deep = Predict(NN_deep.forward(X_test))

print(predict_NN_deep[:,:5])
data = {

    "PassengerId": np.squeeze(Y_test).tolist(),

    "Survived": np.squeeze(predict_NN_deep).tolist(),

}

results = pd.DataFrame(data,columns=["PassengerId","Survived"])

results.to_csv('../working/results.csv',index=False)

results.head()