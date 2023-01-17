import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv", delimiter=',')
df.head(10)   # Shwing first 10 rows 
df.info()   # Give information about dataset
df.drop(["Unnamed: 32","id"],axis=1,inplace = True)   # Dropout unncessary column and using implace for saving to df
# We have target feature = 'diagnosis' in order to classification so we split this feature

y = df.diagnosis 
x = df.drop('diagnosis', axis = 1)
ax = sns.countplot(y,label="Count") 
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
x.describe()
y = df.diagnosis.values
x_df = df.drop(["diagnosis"],axis=1)
# X Dataframe should be normalized to avoid dominance among numerical values because it has several features and 
#model success becomes more realistic if numbers are drawn between 0-1.

x = (x_df - np.min(x_df))/(np.max(x_df)-np.min(x_df)).values           # Formula of normalization
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
# Transforming arrays to transpoze in order to avoid getting shape error
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
name_list = ['Logistic Regression', 'Knn(n=8)', 'SVM', 'Native Bayes','Decision Tree', 'Random Foerest']
pred_score = []    # In order to compare all pred scores
# We determite to parameter initialize values for sigmoid function 
# Giving dimension value to funciton and change weight and bias values each step

def change_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

# Creating sigmoid function 

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))    # Sigmoid function formula
    
    return y_head

# Making forward and backward iteration formula functions and changing values each step by step
# We need x_train, y_train and z, loss, cost formula 
# Backward interation contains derivate weight and bias values step by step and return this values with gradients dictionary 
# Evaluating cost and gradients value after these interations

def forward_backward(w,b,x_train,y_train):
    
    z = np.dot(w.T,x_train) + b   #  Sigmoid function formula
    y_head = sigmoid(z)           # Creating pred values to return sigmoid funciton
    
    loss = ((-1)* y_train * np.log(y_head)) + ((-1) * (1 - y_train) * np.log(1- y_head))    # Evaluating loss value within this formula 
    
    #loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    
    cost = (np.sum(loss))/x_train.shape[1]                         # Evaluating cost value within this formula 


    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients

# Each step, updating learning parametres automatically
# Giving learning rate and number of interation value because they are hyperparametre
# Saving cost values in cost_list after confuse cost values step by step

def update(w, b, x_train, y_train, learning_rate, number_iteration):
    
    cost_list = []
    
    for i in range(number_iteration):
        
        cost,gradients = forward_backward(w,b,x_train,y_train)   # Calling previous function
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        # Updating values 
        
    parameters = {"weight": w,"bias": b}
    
    # Show all of them
    
    plt.plot(index,cost_list)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients, cost_list

# Creating predict values and need weight, bias and x_test value
# x_test is a input for forward iteration
# Using sigmoid functions : if z is bigger than 0.5, y_head = 1 or if z is smaller than 0.5, y_head = 0 

def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1])),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1

    return y_prediction
# We use all funcitons for logistic regression 
# We need x and y splits, learning rate and number of iterations
# Initializing value of dimension is x_train.shape[0] 

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, number_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = change_weights_and_bias(dimension)     # Calling change of this values funciton
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,number_iterations)  # Calling update function
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)  # Calling predict function
    
    # Print accuracy value 
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
# Then use logistic regression funciton for example learning rate is 1 and number of iterations are 250


# logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, number_iterations = 250) 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)

print("accuracy of logistic regression is:  {}".format(lr.score(x_test.T,y_test.T)))

pred_score.append(lr.score(x_test.T,y_test.T))

df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv", delimiter=',')
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.tail(10)   # Controlling last 10 rows
df.info()
df.describe()
# Target Feature is diagnosis wthin M and B class

M = df[df.diagnosis == "M"]
B = df[df.diagnosis == "B"]


plt.scatter(M.perimeter_mean,M.texture_mean,color="blue",label="M class diagnosis")
plt.scatter(B.radius_mean,B.texture_mean,color="red",label="B class diagnosis")

plt.xlabel("radius_mean")
plt.ylabel("perimeter_mean")
plt.legend()

plt.show()
plt.scatter(M.radius_mean,M.area_mean,color="green",label="M class diagnosis")
plt.scatter(B.radius_mean,B.area_mean,color="yellow",label="B class diagnosis")

plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.legend()

plt.show()
plt.scatter(M.smoothness_mean,M.compactness_mean,color="cyan",label="M class diagnosis")
plt.scatter(B.smoothness_mean,B.compactness_mean,color="black",label="B class diagnosis")

plt.xlabel("smoothness_mean")
plt.ylabel("compactness_mean")
plt.legend()

plt.show()
sns.jointplot(x='smoothness_mean',y='compactness_mean',data=df,kind='scatter');
sns.jointplot(x='radius_mean',y='perimeter_mean',data=df,kind='scatter');
df.corr()
sns.heatmap(df.corr());

# We assign 1 value if diagnosis is M and we assing 0 value if diagnosis is B  

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]

y = df.diagnosis.values
x_df = df.drop(["diagnosis"],axis=1)
# normalization 
x = (x_df - np.min(x_df))/(np.max(x_df)-np.min(x_df))
# Train and test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

# Using KNN Model

from sklearn.neighbors import KNeighborsClassifier
n_neighbors = 5  # as an example

knn = KNeighborsClassifier(n_neighbors = n_neighbors) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(n_neighbors, knn.score(x_test,y_test)))

# Using KNN Model

n_neighbors = 8  # as an example

knn = KNeighborsClassifier(n_neighbors = n_neighbors) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(n_neighbors, knn.score(x_test,y_test)))

pred_score.append(knn.score(x_test,y_test))

# Finding optimal  k value

score_list = []

for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
from sklearn.svm import SVC
 
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
 
print("accuracy of svm value is: ",svm.score(x_test,y_test))

pred_score.append(svm.score(x_test,y_test))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
 
print("accuracy of naive bayes : ",nb.score(x_test,y_test))

pred_score.append(nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("accuracy of decision tree: ", dt.score(x_test,y_test))

pred_score.append(dt.score(x_test,y_test))


from sklearn.ensemble import RandomForestClassifier

n_estimators = 100  # as an example number of trees

rf = RandomForestClassifier(n_estimators = n_estimators,random_state = 1)

rf.fit(x_train,y_train)

print("accuracy of random forest (100): ",rf.score(x_test,y_test))

pred_score.append(rf.score(x_test,y_test))
y_pred = rf.predict(x_test)
y_true = y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

# visualization with heatmap
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,fmt = ".0f",ax = ax)

plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

accuracy = pd.DataFrame({'algorithmas' : name_list, 'accuracy_value': pred_score})

plt.figure(figsize=(12,6))

plt.plot(accuracy.algorithmas,accuracy.accuracy_value)

plt.title("Comparison of Accuracy Values")

plt.show()
