import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import pandas as pd
torch.cuda.is_available()
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")



from sklearn.preprocessing import StandardScaler



std_scaler = StandardScaler()

    

std_time = std_scaler.fit_transform(df.loc[:, "Time"].values.reshape((-1, 1)))

std_amount = std_scaler.fit_transform(df.loc[:, "Amount"].values.reshape((-1, 1)))



df["Time"] = std_time

df["Amount"] = std_amount



from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X_train = df_train.drop("Class", axis=1).values

y_train = df_train["Class"].values



X_test = df_test.drop("Class", axis=1).values

y_test = df_test["Class"].values
class NeuralNetwork(nn.Module):

    

    def __init__(self, input_size, hidden_sizes, output_size, activation):

        """

        Abstract class for Neural Network model.

        

        Parameters

        ----------

        input_size (int): dimension of input features.

        hidden_sizes (tuple, list): a sequence of layers, each element is the number neurons of that layer.

        output_size (int): dimension of output.

        activation (string): non-linear activation to use. [sigmoid, relu or tanh]

        """

        super().__init__()

        assert activation in ["sigmoid", "relu", "tanh"]

        self.hidden_sizes = hidden_sizes

        if activation == "sigmoid":

            self.activation = nn.Sigmoid()

        elif activation == "relu":

            self.activation = nn.ReLU()

        else:

            self.activation = nn.Tanh()

        self.net = []

        hidden_sizes = (input_size,) + hidden_sizes

        for i, hidden in enumerate(hidden_sizes[:-1]):

            self.net.append(nn.Linear(in_features=hidden, out_features=hidden_sizes[i+1]))

            self.net.append(self.activation)

        self.net.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))

        self.net = nn.Sequential(*self.net)

    

    def forward(self, x):

        return self.net(x)

    

    def predict(self, x_new):

        with torch.no_grad():

            y_hat = self.net(x_new)

        y_hat = torch.softmax(y_hat, dim=-1)

        pred = torch.argmax(y_hat, dim=-1)

        return pred
class Trainer:

    

    def __init__(self, model, batch_size, epochs, optimizer, loss_func):

        self.model = model

        self.batch_size = batch_size

        self.epochs = epochs

        self.optimizer = optimizer

        self.loss_func = loss_func

        

    def train(self, X_train, y_train):

        m = X_train.shape[0]

        num_batches = int(np.ceil(m/self.batch_size))

        

        for e in range(self.epochs):

            indices = torch.randperm(m)

            X_train = X_train[indices]

            y_train = y_train[indices]

            epoch_loss = 0.0

            

            for b in range(num_batches):

                X_batch = X_train[b*self.batch_size:(b+1)*self.batch_size]

                y_batch = y_train[b*self.batch_size:(b+1)*self.batch_size]

                

                y_hat = self.model(X_batch)

                batch_loss = self.loss_func(y_hat, y_batch)

                

                self.optimizer.zero_grad()

                batch_loss.backward()

                self.optimizer.step()

                

                epoch_loss += batch_loss.item()

        

        

            print("Loss at epoch %d: %.5f" % (e, epoch_loss/num_batches))
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support



class Evaluator:

    

    def __init__(self, model, loss_func):

        self.model = model

        self.loss_func = loss_func

    

    def evaluate(self, X_test, y_test):

        print("-"*50)

        print("Neural Network - Hidden layer sizes: %s - Activation: %s" % (str(self.model.hidden_sizes), str(self.model.activation)))

        y_pred = self.model.predict(X_test)

        with torch.no_grad():

            loss = self.loss_func(self.model(X_test), y_test)

        print("Testing Loss: %.5f" % loss)

        conf_mat = confusion_matrix(y_test, y_pred)

        print(conf_mat)

        prf = precision_recall_fscore_support(y_test, y_pred, average="binary")

        print("Accuracy: %.5f" % ((conf_mat[0, 0] + conf_mat[1, 1])/conf_mat.sum() ))

        print("Precision: %.5f" % prf[0])

        print("Recall: %.5f" % prf[1])

        print("F1-score: %.5f" % prf[2])
def _11(X_train, y_train, X_test, y_test):

    # Using random seed to make sure the program is preproducibility

    torch.manual_seed(12)

    np.random.seed(12)



    # Initialize neural network object to training

    neural_network = NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=(50, 32, 16), output_size=2, activation="relu")

    print(neural_network.parameters)

    

    # Define loss function and optimizer to optimize the neural network parameters

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.001, betas=(0.9, 0.999))

    

    # 2 helper classes: Trainer and Evaluator

    trainer = Trainer(model=neural_network, batch_size=256, epochs=5, optimizer=optimizer, loss_func=loss_func)

    evaluator = Evaluator(model=neural_network, loss_func=loss_func)

    

    X_train = torch.Tensor(X_train)

    y_train = torch.Tensor(y_train).long()

    trainer.train(X_train, y_train)

    

    X_test = torch.Tensor(X_test)

    y_test = torch.Tensor(y_test).long()

    evaluator.evaluate(X_test, y_test)
_11(X_train, y_train, X_test, y_test)
def _12(X_train, y_train, X_test, y_test, weights):

    # Using random seed to make sure the program is preproducibility

    torch.manual_seed(12)

    np.random.seed(12)

    

    # Initialize neural network object to training

    neural_network = NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=(50, 32, 16), output_size=2, activation="relu")

    print(neural_network.parameters)

    

    # Define loss function and optimizer to optimize the neural network parameters

    loss_func = nn.CrossEntropyLoss(torch.Tensor(weights))

    optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.001, betas=(0.9, 0.999))



    # 2 helper classes: Trainer and Evaluator

    trainer = Trainer(model=neural_network, batch_size=256, epochs=5, optimizer=optimizer, loss_func=loss_func)

    evaluator = Evaluator(model=neural_network, loss_func=loss_func)

    

    X_train = torch.Tensor(X_train)

    y_train = torch.Tensor(y_train).long()

    trainer.train(X_train, y_train)

    

    X_test = torch.Tensor(X_test)

    y_test = torch.Tensor(y_test).long()

    evaluator.evaluate(X_test, y_test)
_12(X_train, y_train, X_test, y_test, weights=[0.1, 0.9])
_12(X_train, y_train, X_test, y_test, weights=[0.01, 0.99])
class FocalLoss(nn.Module):

    

    def __init__(self, gamma=2, alpha=1):

        super().__init__()

        self.gamma = gamma

        self.alpha = alpha

        

    def forward(self, y_hat, y):

        """

        Parameters

        ----------

        y_hat: un-normalized, raw output of model (output from linear layer). shape = (N, C)

        y: 1D ground truth label, shape = (N,)

        """

        if type(self.alpha) is list:

            self.alpha = torch.Tensor(self.alpha)

        N, C = y_hat.shape

        y = y.view(-1, 1)

        y_onehot = torch.FloatTensor(N, C)



        y_onehot.zero_()

        y_onehot.scatter_(1, y, 1)



        y_hat = torch.softmax(y_hat, dim=-1)

        

        loss = -y_onehot * self.alpha * (1-y_hat)**self.gamma * torch.log(y_hat)

        return torch.mean(torch.sum(loss, dim=-1))
def _13(X_train, y_train, X_test, y_test, alpha, gamma):

    # Using random seed to make sure the program is preproducibility

    torch.manual_seed(12)

    np.random.seed(12)

    

    # Initialize neural network object to training

    neural_network = NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=(50, 32, 16), output_size=2, activation="relu")

    print(neural_network.parameters)

    

    # Define loss function and optimizer to optimize the neural network parameters

    loss_func = FocalLoss(gamma=gamma, alpha=alpha)

    optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.001, betas=(0.9, 0.999))



    # 2 helper classes: Trainer and Evaluator

    trainer = Trainer(model=neural_network, batch_size=256, epochs=5, optimizer=optimizer, loss_func=loss_func)

    evaluator = Evaluator(model=neural_network, loss_func=loss_func)

    

    X_train = torch.Tensor(X_train)

    y_train = torch.Tensor(y_train).long()

    trainer.train(X_train, y_train)

    

    X_test = torch.Tensor(X_test)

    y_test = torch.Tensor(y_test).long()

    evaluator.evaluate(X_test, y_test)
alphas = [[1, 1], [0.1, 0.9], [0.3, 0.7]]

gammas = [1, 2, 3, 4, 5]



print("Do grid search for hyperparameters of Focal Loss: alpha, gamma")

for a in alphas:

    for g in gammas:

        print("Alpha: %s, Gamma: %s" % (str(a), str(g)))

        _13(X_train, y_train, X_test, y_test, alpha=a, gamma=g)

        print("*"*80)
_13(X_train, y_train, X_test, y_test, alpha=[0.1, 0.9], gamma=3) # work best, from the grid search above
import xgboost as xgb
def binary_cross_entropy(y:np.ndarray, y_hat: np.ndarray):

    '''Binary Cross Entropy objective. 

    '''

    grad = (y_hat - y) / (y_hat * (1 - y_hat))

    hess = (y*(1-2*y_hat) + y_hat*(3-y_hat)) / (y_hat*(1-y_hat))**2

    return grad, hess
def _21(X_train, y_train, X_test, y_test):

    params = {"booster": "gbtree", "max_depth": 5, "learning_rate": 0.4, "n_estimators": 100, "seed": 12, 

              "num_classes": 2, "objective": binary_cross_entropy, "verbosity": 1, "random_state": 13}

    clf = xgb.XGBClassifier()

    clf.set_params(**params)

    print(clf)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred[y_pred >= 0.5] = 1

    y_pred[y_pred < 0.5] = 0



    print("-"*50)

    conf_mat = confusion_matrix(y_test, y_pred)

    print(conf_mat)

    prf = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print("Accuracy: %.5f" % ((conf_mat[0, 0] + conf_mat[1, 1])/conf_mat.sum() ))

    print("Precision: %.5f" % prf[0])

    print("Recall: %.5f" % prf[1])

    print("F1-score: %.5f" % prf[2])
_21(X_train, y_train, X_test, y_test)
def weighted_binary_cross_entropy(y:np.ndarray, y_hat: np.ndarray):

    """

    Weighted Binary Cross Entropy objective.

    """

    alpha_1 = 0.99 # for class 1

    alpha_0 = 0.01 # for class 0

    grad = (y*y_hat*(alpha_0-alpha_1) + alpha_0*y_hat - alpha_1*y) / (y_hat * (1 - y_hat))

    hess = (y_hat**2*(y*(alpha_0 - alpha_1) + 3*alpha_0) + y*(alpha_1 - 2*alpha_1*y_hat)) / (y_hat*(1-y_hat))**2

    return grad, hess
def _22(X_train, y_train, X_test, y_test):

    params = {"booster": "gbtree", "max_depth": 5, "learning_rate": 0.4, "n_estimators": 100, "seed": 12, 

              "num_classes": 2, "objective": weighted_binary_cross_entropy, "verbosity": 1, "random_state": 13}

    clf = xgb.XGBClassifier()

    clf.set_params(**params)

    print(clf)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred[y_pred >= 0.5] = 1

    y_pred[y_pred < 0.5] = 0



    print("-"*50)

    conf_mat = confusion_matrix(y_test, y_pred)

    print(conf_mat)

    prf = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print("Accuracy: %.5f" % ((conf_mat[0, 0] + conf_mat[1, 1])/conf_mat.sum() ))

    print("Precision: %.5f" % prf[0])

    print("Recall: %.5f" % prf[1])

    print("F1-score: %.5f" % prf[2])
_22(X_train, y_train, X_test, y_test)
from imblearn.over_sampling import SMOTE



smote = SMOTE(random_state=13)



X_train, y_train = smote.fit_sample(X_train, y_train)
y_train.shape
_11(X_train, y_train, X_test, y_test)
_12(X_train, y_train, X_test, y_test, weights=[0.1, 0.9])
_12(X_train, y_train, X_test, y_test, weights=[0.01, 0.99])