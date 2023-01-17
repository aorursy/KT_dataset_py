#import library

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

import pandas as pd

import matplotlib.pyplot as plt
#load dataset

data = pd.read_csv('../input/indian_liver_patient.csv')
#view dataset

data.head()
#visualization age-albumin and globulin ratio

plt.scatter(data.Age, data.Albumin_and_Globulin_Ratio, color = 'navy',alpha = 0.3)

plt.xlabel('Age')

plt.ylabel('Albumin_and_Globulin_Ratio')

plt.title('Visualization for Age-Albumin_and_Globulin_Ratio')

plt.show()
data.info()
corr = data.corr()

corr.style.background_gradient()
# we created x and y array. So we understand very well.

y = np.array([2,5,10,14,15,16,20,25,30,35,36,38,40,45,50,52,55,60,61,62]).reshape(-1,1)

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]).reshape(-1,1)
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x,y)

b_zero = linear_reg.intercept_

b_one = linear_reg.coef_

print('b_zero: {}, b_one: {}'.format(b_zero,b_one))

#visualization

plt.scatter(x,y)

x_predict = linear_reg.predict(x)

plt.plot(x,x_predict,color='red')
#Now we visualization for dataset

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = data.Total_Bilirubin.values.reshape(-1,1)

y = data.Direct_Bilirubin.values.reshape(-1,1)

linear_reg.fit(x,y)

b_zero = linear_reg.intercept_

b_one = linear_reg.coef_

print('b_zero: {}, b_one: {}'.format(b_zero,b_one))

#visualization

plt.scatter(x,y)

x_predict = linear_reg.predict(x)

plt.plot(x,x_predict,color='red')
# This regression may involve x^2,x^3,....

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



y = np.array([100,95,93,90,86,85,80,82,75,70,65,55,40,45]).reshape(-1,1)

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14]).reshape(-1,1)

plt.scatter(x,y)

polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regressionn = LinearRegression()

linear_regressionn.fit(x_polynomial,y)

y_head2 = linear_regressionn.predict(x_polynomial)

plt.plot(x,y_head2,color = 'purple',label="ploy")

plt.legend()

plt.show()
y = data.Dataset.values.reshape(-1,1)

x_data = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin']]
#normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



# create initialize weights and bias values

def initialize_weights_and_bias(dimension):

    #create weights

    w = np.full((dimension,1),0.01)

    #create bias

    b = 0.0

    return w,b

#w,b = initialize_weights_and_bias(5)

#w,b
# create activation function

def sigmoid(z):

    #sigmoid function

    y_head = 1/(1+np.exp(-z))

    return y_head
#create forward backward propagation

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
# for values update

def update(w,b,x_train,y_train,learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    for i in range(number_of_iterarion):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w = w-learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

       # if block created for visualization

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

            

    parameters = {"weight":w, "bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index, rotation = 'vertical')

    plt.xlabel("number of iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list

#prediction

def predict(w,b,x_test):

    t = np.dot(w.T,x_test)+b

    z = sigmoid(t)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i] = 2

        else:

            Y_prediction[0,i] = 1

    return Y_prediction

# %% logistic_regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  number_of_iterarion):

    # initialize

    dimension =  x_train.shape[0]  

    w,b = initialize_weights_and_bias(dimension)

    

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,number_of_iterarion)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, number_of_iterarion = 30)    
data1 = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Dataset']]
data1.head()
#patient has liver disease or not. One =liver disease, two =liver disease not.

one = data1[data1.Dataset == 1]

two = data1[data1.Dataset == 2]
plt.scatter(one.Age, one.Total_Protiens, color = "purple", label = "one", alpha = 0.4)

plt.scatter(two.Age, two.Total_Protiens, color = "orange", label = "two", alpha = 0.4)

plt.xlabel("Age")

plt.ylabel("Total_Protiens")

plt.legend()

plt.show()
y = data.Dataset.values.reshape(-1,1)

x_data = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin']]
# normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# (x-minx)/(maxx-minx)
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print("{} nın score: {}".format(2,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list,color='red')

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
y = data.Dataset.values.reshape(-1,1)

x_data = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin']]

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#svm

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("svm algoritmasının doğruluğu: ",svm.score(x_test, y_test))
y = data.Dataset.values.reshape(-1,1)

x_data = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin']]

#normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

#navie bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

print("bayes algoritmasının sonucu: ", nb.score(x_test, y_test))
y = data.Dataset.values.reshape(-1,1)

x_data = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin']]

#normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

print("score: ", dt.score(x_test, y_test))
y = data.Dataset.values.reshape(-1,1)

x_data = data[['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin']]

#normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 10, random_state = 1)

rf.fit(x_train,y_train)

print("random forest result: ", rf.score(x_test,y_test))

#confusion matrix

y_pred = rf.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

#visualization

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("prediction values")

plt.ylabel("original values")

plt.show()