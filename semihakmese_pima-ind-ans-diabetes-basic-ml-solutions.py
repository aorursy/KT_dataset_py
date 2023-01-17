# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading Data

data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
#Let's see the columns

data.columns
# then we need to see info

data.info()
data.head()
#Let's Visualize the Data 

color_list = ["red" if i == 1 else "green" for i in data.loc[:,"Outcome"]]

pd.plotting.scatter_matrix(data.loc[:,data.columns !="Outcome"],

                          c=color_list,

                          figsize = [20,20],

                          diagonal ="hist",

                          alpha = 0.6,

                          s=200,

                          marker ="*",

                          edgecolor = "black")

plt.show()
#To see the distrubution of the outcome we'll use seaborn sns.countplot

import seaborn as sns

sns.countplot(x="Outcome", data = data)

data.loc[:,"Outcome"].value_counts()

y = data.Outcome.values

x_data = data.iloc[:,:-1]

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state = 42) #Yüzde 25 i x_test vey_test e atanacak 



x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T



print("x_train shape",x_train.shape)

print("y_train shape",y_train.shape)

print("x_test shape",x_test.shape)

print("y_test shape",y_test.shape)
#%% Initializing Parameters and Sigmoid Function 



def initialize_weights_and_bias(dimension): #30 feature var o zaman 30 dimension olmalı

    

    w = np.full((dimension,1),0.01) #burada dimension 30 girdiğimiz zaman [0,0.01] lik weightler atayacagız

    b = 0.0 #float olsun diye 0.0 yazdım

    return w,b

# w,b = initialize_weight_and_bias(30)



def sigmoid(z):

    y_head = 1/(1+ np.exp(-z)) #formülü budur z nin

    return y_head



#sigmoid(0) değeri 0.5 vermelidir 

    
#%% Forward - Backward Propagation

#Bu kısımda w ile train data mızı çarpacağız Bias ekleyip sigmoid fonksiyona sokacağız 



def forward_backward_propagation(w,b,x_train,y_train):

    #Forward Prop 

    z = np.dot(w.T,x_train) + b #Transpoz alma sebebimiz matris carpımını yapabilmek için 

    y_head = sigmoid(z) #Sigmoid fonksiyonuna soktuk

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #Loss fonksiyonunu yazdık 

    cost = (np.sum(loss)) / x_train.shape[1] #Losslar toplamını normalize etmek için sample sayısına böldük 

    #x_train_shape[1] = 455

    



    #Backward Prop

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] #Formül bu, shape bölmek normalize etmek için

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
#%% Updating Parameters 



def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    #Iteration 

    for i in range(number_of_iterarion):

        #Doing forward and Backward Propagation 

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost) #Güncelleme öncesi cost list e atıyorum (Tüm cost listleri depolamak)

        

        #Updating 

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

    

        if i %10 == 0:

            cost_list2.append(cost) #Her 10 adımda bir costları depola    

            index.append(i)

            print("Cost after iteration %i: %f"%(i,cost))

            

    #Number of iteration kaç olacagı kararını deneyerek bulacagız Türevi 0 a yaklaşınca yeterli olacaktır

    #We updaate (learn) parameters weights and Bias 

    parameters = {"weight":w , "bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation ='vertical')

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters,gradients,cost_list
#%% Prediction 

def predict(w,b,x_test): #w,b zaten lazım ama x_test de class ı belli olmayan ve test edeceğim (tahmin edeceğim) data 

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    

    #Eger z 0.5 den büyük ise y_head = 1 yani kötü huylu

    #Eger < 0.5 ise y_head = 0 yani iyi huylu 

    

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

            

    return y_prediction

        

#şimdi y prediction u y test ile karşılastırıp eğitimin dogruluguna bakıcaz 

    
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 400) 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading Data

data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data0 = data[data.Outcome == 0] # Healthy group

data1 = data[data.Outcome == 1] # Sick group

data1.sort_values(by="Age")
#We will use BloodPressure and Age parameters in the Sick group

data1 = data[data.Outcome == 1]

xlin=data1.BloodPressure.values.reshape(-1,1)

ylin=data1.Age.values.reshape(-1,1)



#Linear Regression Model

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

#Creating Prediction Space to get more efficient results

predict_space = np.linspace(min(xlin),max(xlin)).reshape(-1,1)  

#Fit

linear_reg.fit(xlin,ylin)

#Prediction 

predicted = linear_reg.predict(predict_space)

#Perfomance Analysis w/R^2 Score method

print("R^2 Score is :",linear_reg.score(xlin,ylin))



plt.plot(predict_space, predicted, color="black",linewidth=2)

plt.scatter(x=xlin, y=ylin)

plt.xlabel("Blood Pressure")

plt.ylabel("Age")

plt.show()
#We will use Glucose and Age parameters in the Sick group

data1 = data[data.Outcome == 1]

xpol=data1.Glucose.values.reshape(-1,1) #In Sklearn we need to reshape our data like that

ypol=data1.Age.values.reshape(-1,1)



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 3) #degree = 3 means we have limited the equation with x^3

x_polynomial = poly_reg.fit_transform(xpol) #We transformed our xpol values to x^3

#Fit (For fitting we use Linear Regression again..)

linear_reg2 = LinearRegression()  

linear_reg2.fit(x_polynomial,ypol)

#Prediction 

y_head = linear_reg2.predict(x_polynomial)

#Visualisation 

plt.plot(x_polynomial,y_head,color="red")

plt.show()
#We will use BloodPressure and Age parameters in the Sick group

data1 = data[data.Outcome == 1]

xdt=data1.BloodPressure.values.reshape(-1,1)

ydt=data1.Age.values.reshape(-1,1)



from sklearn.tree import DecisionTreeRegressor

dtreg = DecisionTreeRegressor()

#Fit

dtreg.fit(xdt,ydt) 

#Prediction space

xdt_ = np.arange(min(xdt),max(xdt),0.01).reshape(-1,1)

y_headdt = dtreg.predict(xdt_)

#Visualisation 

plt.scatter(xdt,ydt,color ="red",label="Values")

plt.plot(xdt_,y_headdt,color="blue",label="Predicted")

plt.show()
#Let's work same features in the Decision Tree Regression model that BloodPressure and Age 

data1 = data[data.Outcome == 1]

xrf=data1.BloodPressure.values.reshape(-1,1)

yrf=data1.Age.values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor  

rfreg = RandomForestRegressor(n_estimators = 100, #We work with 100 times decision tree reg

                             random_state= 42)

#Fit

rfreg.fit(xrf,yrf) 

#Prediction space

xrf_ = np.arange(min(xrf),max(xrf),0.01).reshape(-1,1)

y_headrf = rfreg.predict(xdt_)

#Visualisation 

plt.scatter(xrf,yrf,color ="red",label="Values")

plt.plot(xrf_,y_headrf,color="blue",label="Predicted")

plt.show()
y = data.Outcome.values

x_data = data.iloc[:,:-1]

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state = 42) #Yüzde 25 i x_test vey_test e atanacak 





print("x_train shape",x_train.shape)

print("y_train shape",y_train.shape)

print("x_test shape",x_test.shape)

print("y_test shape",y_test.shape)



#When we need to do any process without any mistake, error arrays should be like that 

# 537,8 

# 537,

# 231,8

# 231
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("Test Accuracy : %{}".format(lr.score(x_test,y_test)*100))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 40) #n_neighbors is a hyperparameter that's why we need to try to examine the Optimum value 

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

# print("Prediction:",prediction) if you want to compare the test data and predictions you can remove # and try 

print("for n={} KNN Score : {}".format(40,knn.score(x_test,y_test))) 

#if we want to see which n number will be optimum we can define a for loop for that 

score_list = []

for each in range(1,50):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test, y_test))



plt.figure(figsize=(8,5))

plt.scatter(range(1,50),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show() 



#40 might be the optimum number for N 

#The answer is 0.753
neig = np.arange(1,50)

train_accuracy = []

test_accuracy = []

#Loop all over in k values

for i,k in enumerate(neig):

    knn3 = KNeighborsClassifier(n_neighbors = k)

    #Fit process

    knn3.fit(x_train,y_train)

    #Train Accuracy

    train_accuracy.append(knn3.score(x_train,y_train))

    #Test Accuracy

    test_accuracy.append(knn3.score(x_test,y_test))

    

#Plotting the Values 

plt.figure(figsize =(13,10))

plt.plot(neig, test_accuracy, label = "Testing Accuracy")

plt.plot(neig, train_accuracy, label = "Training Accuracy")

plt.legend()

plt.title("Values vs Accuracy")

plt.xlabel("Number of Neighbors")

plt.ylabel("Accuracy")

plt.xticks(neig) #We limit the Max min values in the plot axis according to Number of max neighbor

plt.savefig("graph.png")

plt.show()

print("Best Accuracy : {} with K : {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
from sklearn.svm import SVC 

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)



#Test 

print("Accuracy of the SVM Algorithm : ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB 

nb = GaussianNB()

nb.fit(x_train,y_train)



print("Accuracy of the Naive Bayes Algorithm :",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

print("Accuracy of the Decision Tree :",dt.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 24, random_state=42)

rf.fit(x_train,y_train)

print("Accuracy of the Random Forest Classification : ",rf.score(x_test,y_test))



#We need to find optimum value that's why need to decide best number for n_estimators parameter

#if we want to see which n number will be optimum we can define a for loop for that 

score_list2 = []

for each in range(1,200):

    rf2 = RandomForestClassifier(n_estimators = each)

    rf2.fit(x_train, y_train)

    score_list2.append(rf2.score(x_test, y_test))



plt.figure(figsize=(8,5))

plt.scatter(range(1,200),score_list2)

plt.xlabel("n values")

plt.ylabel("accuracy")

plt.show() 
aa = np.max(score_list2) #We can see the max value would be 0.7878 and n = 24 might be the great option

aa
print("Test Accuracy for Logistic Regression: %{}".format(lr.score(x_test,y_test)*100))

print("for n={} KNN Score : %{}".format(40,knn.score(x_test,y_test)*100))

print("Accuracy of the SVM Algorithm : %{}".format(svm.score(x_test,y_test)*100))

print("Accuracy of the Naive Bayes Algorithm : %{}".format(nb.score(x_test,y_test)*100))

print("Accuracy of the Decision Tree : %{}".format(dt.score(x_test,y_test)*100))

print("Accuracy of the Random Forest Classification : %{}".format(rf.score(x_test,y_test)*100))
# I'm gonna show an example on Random Forest C. model 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 24, random_state=42)

rf.fit(x_train,y_train)

print("Accuracy of the Random Forest Classification : ",rf.score(x_test,y_test))



#In this method we need to predict x test values 

y_pred = rf.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



#Lets Visualize it 

import seaborn as sns 

f,ax =plt.subplots(figsize=(6,6))

sns.heatmap(cm, annot = True, linecolor = "blue", ax = ax)



plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
