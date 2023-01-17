import pandas as pd                  #importing the library for data manipulation and storage 

import numpy as np                   #importing the library for scientific computation

import seaborn as sns                #importing the library used for interactive plots

import matplotlib.pyplot as plt      #importing the library used for plots (not high end)

from sklearn.metrics import classification_report, confusion_matrix   #importing sub components from sklearn library

from sklearn.model_selection import train_test_split   #importing train_test_split which we would use later  
df = pd.read_csv('../input/heart.csv')   #reading the csv from the directory and storing the values in df
df.head()   #having a look at the first 5 rows of the dataframe
df['target'].value_counts()     #counting the number of target variables (the number of diseased vs non-diseased heart people)
plt.scatter(x = 'chol', y = 'trestbps', color = 'green', data = df)  #using scatter plot to see the relationship between cholosterol and trestbps

plt.xlabel('cholestrol')    #giving the label to the x-axis                

plt.ylabel('trestbps')      #giving the label to the y-axis

plt.title('Cholestrol Vs Trestbps')  #giving a name to the title
df.columns                #having a look at various input features or columns 
plt.scatter(x = 'chol', y = 'age', color = 'orange', data = df)   #using scatter plot, noticing the relationship between cholestrol and age

plt.xlabel('Cholestrol')        #labeling the x-axis 

plt.ylabel('Age')               #labeling the y-axis

plt.title('Cholestrol vs Age')  #giving the title to the graph
sns.countplot(df['sex'], palette = ("RdYlGn"))           #counting the number of male and female candidates in the dataset
sns.distplot(df['age'])    #having a look at the distribution of males and females in the plot
sns.boxplot(x = 'age', palette = "BuGn", data = df)  #using box plot to see how the age is distributed
sns.jointplot(x = 'age', y = 'oldpeak', kind = 'kde', color = 'Gold', data = df)
sns.jointplot(x = 'cp', y = 'age', kind = 'kde', color ='Green', data = df)
df.head(1)         #having a look at the dataset again

X = df.drop(['target'], axis = 1)      #here, we would take all the columns except 'target' as input vector

y = df['target']                       #here, we are taking the output as the 'target' column in our dataset

ynewtest = y

xnewtest = X

y = y[:, np.newaxis]                   #converting the output to an array 

print('The shape of the input is {}'.format(X.shape))     #printing the shape of the input

print('The shape of the output is {}'.format(y.shape))    #printing the shape of the output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)
print('The shape of the input training set is {}'.format(X_train.shape))

print('The shape of the output training set is {}'.format(y_train.shape))

print('The shape of the input testing set is {}'.format(X_test.shape))

print('The shape of the output testing set is {}'.format(y_test.shape))
#We are initially defining the sigmoid function that could be used later

def sigmoid(z):

    

    s = 1 / (1 + np.exp(-z))

    

    return s
#This is a function that is used to initialize the weights with 0 and biases also with 0

def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))

    b = 0

    return w, b
#this network ensures that there is a forward propagation and at the same time, returns the cost

def propagate(w, b, X, y):

    

    m = X.shape[0]

    A = sigmoid(np.dot(X, w) + b)

    cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) #computing the cost function or the error function

    dw = (1 / m) * np.dot(X.T, (A - y))   #this is derivative of the cost function with respect to w

    db = (1 / m) * np.sum(A - y)          #this is the derivative of the cost function with respect to b

    grads = {'dw': dw, 'db': db}          #these values are stored in a dictionary so as to access them later

    return grads, cost 

#We are trying to get the parameters w and b after modifying them using the knowledge of the cost function

def optimize(w, b, X, y, num_iterations, learning_rate, print_cost = False):

    costs = []                    #This is an empty list created so that it stores all the values later

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, y)       #we are calling the previously defined function 

        dw = grads['dw']                          #we are accessing the derivatives of cost with respect to w

        db = grads['db']                          #we are accessing the derivatives of cost with respect to b

        w = w - learning_rate * dw                #we are modifying the parameter w so that the cost would reduce in the long run

        b = b - learning_rate * db                #we are modifying the parameter b so that the cost would reduce in the long run

        np.squeeze(cost)

        if i % 100 == 0:

            costs.append(cost)                    #we are giving all the cost values to the empty list that was created initially

        if print_cost and i % 1000 == 0:

            print("cost after iteration {}: {}".format(i, cost))

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate = " + str(learning_rate))

    plt.show()

    params = {'w': w, 'b': db}                    #we are storing this value in the dictionary so that it could be accessed later

    grads = {'dw': dw, 'db': db}                  #we are storing these valeus in the dictionary so that they could be accessed later

    return params, grads, costs
#This is a function that gives 1 if the activation is greater that 0.5 and 0 otherwise

def predict(w, b, X):

    m = X.shape[0]

    y_prediction = np.zeros((m, 1))

    A = sigmoid(np.dot(X, w) + b)

    for i in range(A.shape[0]):

        if (A[i, 0] <= 0.5):

            y_prediction[i, 0] = 0

        else:

            y_prediction[i, 0] = 1

            

    return y_prediction
def model(X_train, X_test, y_train, y_test, num_iterations, learning_rate, print_cost = True):

    w, b = initialize_with_zeros(X.shape[1])

    parameters, grads, costs = optimize(w, b, X, y, num_iterations, learning_rate, print_cost = True)

    w = parameters["w"]

    b = parameters["b"]

    y_prediction_test = predict(w, b, X_test)

    y_prediction_train = predict(w, b, X_train)

    

    print('train accuracy: {}'.format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print('test accuracy: {}'.format("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)))

    

    d = {"costs": costs,

         "y_prediction_test": y_prediction_test, 

         "y_prediction_train" : y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "num_iterations": num_iterations}

    

    return d
d = model(X_train, X_test, y_train, y_test, num_iterations = 100000, learning_rate = 0.00015, print_cost = True)
df
xpred = xnewtest

ypred = ynewtest

i = 300         #play around with this number to access each row in the training and test set and check the accuracy

xnewpred = xpred.iloc[i]

ynewpred = ypred.iloc[i]

print('The input values of the features are:')

print(xnewpred)

print('The actual output whether a person has a heart disease or not is:')

print(float(ynewpred))

xnewpred = xnewpred[:, np.newaxis]

xnewpred = xnewpred.T

ynew = predict(d["w"], d["b"], xnewpred)

print('The output of the predicted value is:')

print(ynew[0][0])
