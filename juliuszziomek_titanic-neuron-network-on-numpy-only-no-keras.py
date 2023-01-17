import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Data import from the csv file to a dictionary with numpy arrays

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

#Shuffle dataset

train_data = train_data.sample(frac=1)



columns = dict()

for column in train_data.columns:

    columns[column] = np.array(train_data[column]).reshape(-1,1)
from scipy import stats #Needed for convienient mode function



def binarize(vector):

    #Get rid of nan values and replace them with the most common value in the whole feature

    for i in range(len(vector)):

        try: 

            if np.isnan(np.array(vector[i],dtype=np.float64)): vector[i] = stats.mode(vector).mode

        except: 

            pass

    

    #Create new matrix storing the binary endcoded vector

    binarized_matrix = np.zeros(shape=(vector.shape[0],len(np.unique(vector))))

    

    for no,value in enumerate(np.unique(vector)):

        for entry in range(vector.shape[0]):

            if vector[entry] == value: binarized_matrix[entry,no] = 1

        

    #Return binarized matrix without the last column, to avoid the curse of dimensionality

    return binarized_matrix[:,:len(np.unique(vector))-1]

    

def standarize(vector,mean,std):

    #Get rid of nan values and replace them with mean

    vector[np.isnan(vector)==True] = np.nanmean(vector)

    

    #Subtract mean and divide by variance

    vector -= mean

    vector /= std

    

    return vector
#Continuous features are Age and Fare, all of them should be standarized

Age_mean = np.nanmean(columns['Age'])

Age_std = np.nanstd(columns['Age'])

Fare_mean = np.nanmean(columns['Fare'])

Fare_std = np.nanstd(columns['Fare'])

columns['Age'] = standarize(columns['Age'],Age_mean,Age_std)

columns['Fare'] = standarize(columns['Fare'],Fare_mean,Fare_std)



#Discrete features are: Pclass, Sex, SibSp, Parch,Embarked, out of which Pclass, Sex and Embarked should be binarized

columns['Pclass'] = binarize(columns['Pclass'])

columns['Sex'] = binarize(columns['Sex'])

columns['Embarked'] = binarize(columns['Embarked'])



#All of the features mentioned above shall be concatenated into one array:

X = columns['Age']

for column in {'Embarked','Fare','Sex','SibSp','Parch','Pclass'}:

    X = np.concatenate((X,columns[column]),axis=1)



Y = columns['Survived']



#Divide into subsets:

m = X.shape[0]

X_train = X[:(int)(m*0.6),:]

X_cv = X[(int)(m*0.6):(int)(m*0.8),:]

X_test = X[(int)(m*0.8):,:]



Y_train = Y[:(int)(m*0.6),:]

Y_cv = Y[(int)(m*0.6):(int)(m*0.8),:]

Y_test = Y[(int)(m*0.8):,:]
def relu(x):

    x = np.copy(x)

    x[x<0] = 0

    return x



def relu_backward(x):

    x = np.copy(x)

    x[x<0] = 0

    x[x>=0] = 1

    return x



def sigmoid(x):

    return 1/(1+np.exp(-x))



def sigmoid_backward(x):

    return sigmoid(x)*(1-sigmoid(x))
EPSILON = 1e-10 #To avoid dividing by zero and taking logs from negative numbers



#Function for random initialization of weights:

def initialize_weights(m,n):

    w = np.random.randn(m,n)

    return w



class Network:

    def __init__(self,in_size,hidden_size,out_size):

        

        self.in_1 = None #Input to first layer

        self.w1 = initialize_weights(in_size + 1,hidden_size) #Weights of first layer (+1 because of bias unit)

        self.w1_grad = np.zeros_like(self.w1) #Gradients of w1

        self.z_1 = None #Output of first layer

        self.out_1 = None #Output of first layer after activation

        

        self.in_2 = None #Input to second layer

        self.w2 = initialize_weights(hidden_size + 1,out_size) #Weights of second layer (+1 because of bias unit)

        self.w2_grad = np.zeros_like(self.w2) #Gradients of w2

        self.z_2 = None #Output of second layer

        self.out_2 = None #Output of second layer after activation

    

  

    def forward(self,X):

        

        #First layer--------------------------

        self.in_1 = np.concatenate((np.ones(shape=(X.shape[0],1)),X),axis=1) #Add bias unit

        

        self.z_1 = np.dot(self.in_1,self.w1) #matmul

        

        self.out_1 = relu(self.z_1) #relu

        #--------------------------------------

        

        #Second layer--------------------------

        self.in_2 = np.concatenate((np.ones(shape=(self.out_1.shape[0],1)),self.out_1),axis=1) #Add bias unit

        

        self.z_2 = np.dot(self.in_2,self.w2) #matmul

        

        self.out_2 = sigmoid(self.z_2) #sigmoid

        

        #--------------------------------------

        

        return self.out_2



    def compute_cost(self,h,Y):

        #Compute cost based on cross entropy

        

        loss = - (Y * np.log(h+EPSILON) +  (1-Y)*np.log(1-h+EPSILON))/Y.shape[0]

        

        return sum(loss)

    

    def backward(self,h,Y):

        #Backprop through cost:

        delta = - (Y * 1/(h+EPSILON) - (1-Y) * 1/(1-h+EPSILON))/Y.shape[0]

        

        #Second layer----------------------------------------------------

        delta = sigmoid_backward(self.z_2)*delta #Backprop through sigmoid

        

        self.w2_grad = np.dot(self.in_2.T,delta) #Backprop through matmul

        

        delta = np.dot(delta,self.w2.T)[:,1:] #Backprop the gradients to next layer

        #----------------------------------------------------------------

        

        

        #First layer-----------------------------------------------------

        delta = relu_backward(self.z_1)*delta #Backprop through relu

        

        self.w1_grad = np.dot(self.in_1.T,delta) #Backprop through matmul

        

        

        #----------------------------------------------------------------

    

    def update_weights(self,learning_rate):

        self.w1 -= learning_rate * self.w1_grad

        self.w2 -= learning_rate * self.w2_grad
model = Network(9,18,1)

cost = []

cost_cv = []
lr = 1e-3 #Learning rate

maxit = 30000 #Number of iterations



for it in range(maxit):

    h = model.forward(X_train)

    model.backward(h,Y_train)

    model.update_weights(lr)

    

    cost.append(model.compute_cost(h,Y_train))

    cost_cv.append(model.compute_cost(model.forward(X_cv),Y_cv))

    

    #Decrease learning rate after 15K iterations

    if it==15000: lr /=3
import matplotlib.pyplot as plt #Needed for plotting



plt.plot(cost)

plt.plot(cost_cv)

plt.show()
TRESHOLD = 0.5

def accuracy(h,Y):

    r = np.zeros_like(h)

    r[h>TRESHOLD] = 1

    return sum(r==Y)/Y.shape[0]



print('Accuracy on the test set is equal to: ',accuracy(model.forward(X_test),Y_test))
#Importing and preprocessing data in the same way as before

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_columns = dict()

for column in test_data.columns:

    test_columns[column] = np.array(test_data[column]).reshape(-1,1)



#Continuos data need to be standarized using the same mean and std as before

test_columns['Age'] = standarize(test_columns['Age'],Age_mean,Age_std) 

test_columns['Fare'] = standarize(test_columns['Fare'],Fare_mean,Fare_std)



test_columns['Pclass'] = binarize(test_columns['Pclass'])

test_columns['Sex'] = binarize(test_columns['Sex'])

test_columns['Embarked'] = binarize(test_columns['Embarked'])



test = test_columns['Age']

for column in {'Embarked','Fare','Sex','SibSp','Parch','Pclass'}:

    test = np.concatenate((test,test_columns[column]),axis=1)
predicted_scores = model.forward(test)



predicted_labels = np.zeros_like(predicted_scores)

predicted_labels[predicted_scores>TRESHOLD] = 1
index = np.arange(test.shape[0]) + 892

submission = np.concatenate((index.reshape(-1,1),predicted_labels),axis=1)

submission = np.array(submission,dtype=np.int32)
pd.DataFrame(submission,columns=['PassengerId','Survived']).to_csv('Submission.csv',index=False)