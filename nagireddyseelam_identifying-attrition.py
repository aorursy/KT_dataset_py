#Importing Libraries and display the records

import numpy as np

import pandas as pd

#from time import time

from IPython.display import display

from sklearn.metrics import accuracy_score



# Load the dataset

data = pd.read_csv("../input/HR_comma_sep.csv")



# Success - Display the first record

display(data.head(5))
nbr_records = len(data)



# Number of records who left the company

nbr_of_emply_lft = len(data[data.left == 1])



# Number of records who are still working

nbr_of_emply_wrkng = len(data[data.left == 0])



# Percentage who left the company

prcnt_emply_lft = float(nbr_of_emply_lft* 100)/nbr_records



# Check null values in dataset

null_count = data.isnull().values.ravel().sum()



# Print the results

print ("Total number of records: {}" .format(nbr_records))



print ("Total number of employee who have moved on: {}" .format(nbr_of_emply_lft))

print ("Total number of employee who have stayed: {}" .format(nbr_of_emply_wrkng))

print ("Percent of employee who have moved on: {:.2f}%" .format(prcnt_emply_lft))

print ("Total number of null rows in Dataframe: {}" .format(null_count))

print ("\n Datatype as below")

print (data.dtypes)
# Split data into features and target



left_raw = data['left']

features_raw = data.drop('left', axis = 1)
features_raw1=pd.DataFrame()
display(features_raw.head(5))

features_raw1=features_raw
a_row = pd.Series([0.38,0.53,2,157,3,0,0,'sales','low'])

row_df=pd.DataFrame([a_row])
def preprocessing(features_raw):



    # Import sklearn.preprocessing.StandardScaler

    from sklearn.preprocessing import MinMaxScaler



    # Initialize a scaler, then apply it to the features

    features_log_transformed = pd.DataFrame(data = features_raw)

    scaler = MinMaxScaler() # default=(0, 1)

    numerical = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company','Work_accident','promotion_last_5years']



    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])



    # Show an example of a record with scaling applied

    display(features_log_minmax_transform.head(n = 5))

    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

    features_final = pd.get_dummies(features_log_minmax_transform)



    # Encode the 'left_raw' data to numerical values

    #left = pd.get_dummies(left_raw, drop_first = True)

    left = pd.get_dummies(left_raw)

    # Print the number of features after one-hot encoding

    encoded = list(features_final.columns)

    print ("{} total features after one-hot encoding.".format(len(encoded)))



    #  see the encoded feature names

    print (encoded)

    #showing the records after applying ohe-hot encoding 

    display(features_final.head(n = 5))

    return features_final,left
features_final,left=preprocessing(features_raw)
df2=features_final.iloc[[0]]
from sklearn.model_selection import train_test_split



# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features_final, 

                                                    left, 

                                                    test_size = 0.4)



# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape[0]))

print ("Testing set has {} samples.".format(X_test.shape[0]))
def calculate_performance( y_test, y_predict, clf_nm):

    from sklearn import metrics

    from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix

    accuracy = accuracy_score(y_test, y_predict)

    fbeta = fbeta_score(y_test, y_predict,beta=0.5, average='weighted', labels=np.unique(y_predict))

    print("--------%s----------" %(clf_nm))

    print("Model has {:.3f} accuracy"  .format(accuracy*100))

    print("Model has {:.3f} fbeta Score"  .format(fbeta))



    return
class Sigmoid():

    def __call__(self, x):

        return 1 / (1 + np.exp(-x))



    def gradient(self, x):

        return self.__call__(x) * (1 - self.__call__(x))

class MeanSquareError():

    def __init__(self): pass



    def loss(self,y,p):

        return np.sum((y - p)**2) / p.size

    def loss(self, y, p):

        # Avoid division by zero

        p = np.clip(p, 1e-15, 1 - 1e-15)

        l=- y * np.log(p) - (1 - y) * np.log(1 - p)

        print("loss -",l)

        return l



    def acc(self, y, p):

        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))



    def gradient(self, y, p):

        return (y**2+2*p-2*y)/p.size

        

from __future__ import print_function, division

import numpy as np

import math





#from mlfromscratch.utils import train_test_split, to_categorical, normalize, accuracy_score, Plot

#from mlfromscratch.deep_learning.activation_functions import Sigmoid, Softmax

#from mlfromscratch.deep_learning.loss_functions import CrossEntropy



class MLPClassifier():

    """Multilayer Perceptron classifier. A fully-connected neural network with one hidden layer.

    Unrolled to display the whole forward and backward pass.

    Parameters:

    -----------

    n_hidden: int:

        The number of processing nodes (neurons) in the hidden layer. 

    n_iterations: float

        The number of training iterations the algorithm will tune the weights for.

    learning_rate: float

        The step length that will be used when updating the weights.

    """

    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.5):

        self.n_hidden = n_hidden

        self.n_iterations = n_iterations

        self.learning_rate = learning_rate

        self.hidden_activation = Sigmoid()

        self.loss = MeanSquareError()



    def _initialize_weights(self, X, y):

        n_samples, n_features = X.shape

        _, n_outputs = y.shape

        

        # Hidden layer

        #limit   = 1 / math.sqrt(n_features)

        #print(n_features)20

        #print(self.n_hidden)10

        self.W  = np.random.uniform(0, 1, [n_features, self.n_hidden])

        

        #print(type(self.W))

        #self.W=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.95,1,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

        #self.w0 = np.zeros((1,self.n_hidden))

        #self.w0=np.random.uniform(0,1,self.n_hidden)

        #self.w0=[[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1],[0.05,0.1]]

        self.w0=np.random.rand(1, self.n_hidden)        

        # Output layer

        print(type(self.w0))

        #limit   = 1 / math.sqrt(self.n_hidden)

        #print(n_outputs)2

        self.V = np.random.uniform(0,1,[self.n_hidden, n_outputs])

        #self.v0 = np.zeros((1, n_outputs))

        #self.v0=np.random.uniform(0,1,n_outputs)

        self.v0=np.random.rand(1,n_outputs)

        #print(self.w0)

    def fit(self, X, y):



        self._initialize_weights(X, y)



        for i in range(self.n_iterations):



            # ..............

            #  Forward Pass

            # ..............



            # HIDDEN LAYER

            hidden_input = X.dot(self.W) + self.w0

            hidden_output = self.hidden_activation(hidden_input)

            # OUTPUT LAYER

            output_layer_input = hidden_output.dot(self.V) + self.v0

            y_pred = self.hidden_activation(output_layer_input)

            #calculating the error

            #print(type(self.loss.loss(y,y_pred)))

            #print("type of y is ",y.columns[0].shape)

            #print("type of y_pred is ",y_pred.shape)

            #df.loc[:,'B':'F']

            k1=y.loc[:,0].to_numpy()

            b1=y_pred.loc[:,0].to_numpy()

            k2=y.loc[:,1].to_numpy()

            b2=y_pred.loc[:,1].to_numpy()

            #print(calculate_error(k1,b1)+calculate_error(k2,b2))

            #printing the iteration count

            print(i,end=' ')

            #print(self.loss.loss(y,y_pred))

            # ...............

            #  Backward Pass

            # ...............



            # OUTPUT LAYER

            # Grad. w.r.t input of output layer

            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.hidden_activation.gradient(output_layer_input)

            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)

            #print(grad_wrt_out_l_input.shape)

            grad_wrt_out_l_input=np.array(grad_wrt_out_l_input)

            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0,keepdims=True)

            # HIDDEN LAYER

            #print(self.V)



            # Grad. w.r.t input of hidden layer

            #grad_wrt_out_l_input.reshape(self.n_hidden,n_outputs)

            #print(type(grad_wrt_out_l_input))

            #print(type(self.V.T))

            grad_wrt_hidden_l_input = np.dot(grad_wrt_out_l_input,self.V.T) * self.hidden_activation.gradient(hidden_input)

            #grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)

            #grad_wrt_hidden_l_input =np.dot(grad_wrt_out_l_input,self.V.T)

            grad_w = X.T.dot(grad_wrt_hidden_l_input)

            grad_wrt_hidden_l_input=np.array(grad_wrt_hidden_l_input)

            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)



            # Update weights (by gradient descent)

            # Move against the gradient to minimize loss

            #print("shape is",type(grad_v))

            self.V  -= self.learning_rate * grad_v

            self.v0 -= self.learning_rate * grad_v0

            self.W  -= self.learning_rate * grad_w

            self.w0 -= self.learning_rate * grad_w0

        #print(self.W)

        #print(self.w0)

        #print(self.V)

        #print(self.v0)



    

    # Use the trained model to predict labels of X

    def predict(self, X):

        # Forward pass:

        hidden_input = X.dot(self.W) + self.w0

        hidden_output = self.hidden_activation(hidden_input)

        output_layer_input = hidden_output.dot(self.V) + self.v0

        y_pred = self.hidden_activation(output_layer_input)

        

        return y_pred



#from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier

classifier = MLPClassifier(n_hidden=10, n_iterations=500,learning_rate=0.5)



classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

def zeros(y_pred):

    c=0

    for i in range(len(y_pred)):

        if(y_pred[i]==0):

            c+=1

    return c

    
y_test=y_test.values.argmax(axis=1)

y_pred=y_pred.values.argmax(axis=1)
nm='MLPClassifier'

calculate_performance( y_test, y_pred, nm)
def identify_attrition(df2):

    y_pred1 = classifier.predict(df2)

    y_pred1=y_pred1.values.argmax(axis=1)

    if(y_pred1==0):

        print("The employee has the maximum probability to stay in the company")

    else:

        print("The employee has the maximum probability to leave the  company") 

    
#Example  

df=[0.315681,0.265625,0.0,0.285047,0.125,0.0,0.0,0,0,0,0,0,0,0,1,0,0,0,1,0]

identify_attrition(df2)