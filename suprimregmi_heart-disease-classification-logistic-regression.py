# importing all the necessary library and tools



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Logistic Regression Classifier from sklearn

from sklearn.linear_model import LogisticRegression



# Evaluation metrics

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import log_loss

from sklearn.metrics import plot_roc_curve



# Model Evaluation

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
H_D = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
H_D.head(4)
H_D.isna().sum()
H_D.info()
H_D.describe()
H_D["target"].value_counts()
CountNoDisease = len(H_D[H_D.target == 0])

CountHaveDisease = len(H_D[H_D.target == 1])

print("Percentage of Patient With H_D --> {:.2f}%".format((CountNoDisease / (len(H_D.target))*100)))

print("Percentage of Patients Without H_D --> {:.2f}%".format((CountHaveDisease / (len(H_D.target))*100)))
# Visualizing

H_D['target'].value_counts().plot(kind = 'bar', color = ["#990000","lightblue"]);
H_D.count()
H_D["sex"].value_counts()
# Comparing sex with target

pd.crosstab(H_D.target, H_D.sex)
# Visualizing above crosstab

pd.crosstab(H_D.target, H_D.sex).plot(kind = "bar", figsize = (10,6), 

                                     color = ["#900000", "#000070"])

plt.title("Heart_Disease according to Sex")

plt.xlabel("0 = No Disease, 1 = Disease")

plt.ylabel("Number")

plt.legend(["Female","Male"]);
pd.crosstab(H_D.target, H_D.age[H_D.target == 1]) # Age Vs Heart_D
pd.crosstab(H_D.age,H_D.target).plot(kind="bar",figsize=(20,6))

plt.title('Age and No of Patient with heart disease')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(H_D.target, H_D.thalach)  # Thalach = maximun heart rate
# Scatter plot

plt.scatter(H_D.age[H_D.target ==1], H_D.thalach[H_D.target == 1], c = 'black')

plt.xlabel("Age")

plt.ylabel("Max_Heart_Rate_Acheived");
# Checking Age Distribution with boxplot

plt.boxplot(H_D.age);
H_D['sex'].value_counts().plot(kind = "bar", color = ["#000050", "#500000"])

plt.title("Heart Disease according to Sex")

plt.xlabel(" Male                            Female ") 

plt.legend();
H_D.corr()
# Visualizing Correlation in heatmap

cor_mat = H_D.corr()

fig, ax = plt.subplots(figsize = (16,10))

ax = sns.heatmap(cor_mat,

                annot = True,

                linewidth = 0.5,

                fmt = ".2f",

                cmap = "YlGnBu")
def split_data(data, test_ratio):

    np.random.seed(42)

    shuffling_indices = np.random.permutation(len(data))

    test_size = int(len(data)*test_ratio)

    test_indices = shuffling_indices[:test_size] # Setting test index from start to upto test_size

    train_indices = shuffling_indices[test_size:]

    

    return data.iloc[train_indices], data.iloc[test_indices]
train_set,test_set = split_data(H_D, 0.2) # Passing 20% test data ratio to function
train_set.shape, test_set.shape
X_train = train_set.drop(["target"], axis = 1)

y_train = train_set["target"]

X_test = test_set.drop("target", axis = 1)

y_test = test_set["target"]
# No of train and test sets 

len(X_train), len(y_train), len(X_test), len(y_test)
# Instiantiate the model

logistic_model = LogisticRegression()



# Fitting the model

logistic_model.fit(X_train, y_train)



# Evaluating Model

logistic_model.score(X_test, y_test)
logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_model.score(X_test, y_test)
LogisticRegression().get_params().keys()
# Creating a dictionary to pass some of the parameters that LogReg takes

log_reg_grid = {"C": np.logspace(-4,4,20),

               "solver" : ["liblinear"],} #The logspace() function return numbers spaced evenly on a log scale.
# Setup random hyperparameter search for Logistic Regression

randomized_search_log_reg = RandomizedSearchCV(LogisticRegression(),

                               param_distributions = log_reg_grid,

                                              cv = 5,

                                              n_iter = 20,

                                              verbose = True)
# Fitting the hyperparmaeter search model for LogisticRegression

randomized_search_log_reg.fit(X_train, y_train)
# Checking the score

randomized_search_log_reg.score(X_test,y_test), randomized_search_log_reg.best_params_
# Setup hyperparameter search 

#grid_search_logistic_regression(gs_lr)

# Setup logistic regression grid(lrg)

lrg = {"C": np.logspace(-4,4,30),

                "solver": ["liblinear"],

               }



gs_lr = GridSearchCV(LogisticRegression(),

                    param_grid = lrg,

                    cv =5,

                    verbose = True)

gs_lr.fit(X_train, y_train)
gs_lr.score(X_test, y_test), gs_lr.best_params_
y_preds = gs_lr.predict(X_test)

y_preds
# ROC curve

plot_roc_curve(gs_lr, X_test, y_test)
# Confusion Matrix

print(confusion_matrix(y_test, y_preds))
# Visualizing confusion matrix



def plot_confusion_mat(y_test, y_preds):

    fig,ax = plt.subplots()

    ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                    annot = True,

                    cbar = False)

    plt.xlabel("True Label")

    plt.ylabel("False Label")

plot_confusion_mat(y_test, y_preds)
H_D.tail(4) # Let's look our dataset before feature encoding
dummies = pd.get_dummies(H_D[["cp", "slope", "thal"]]) # Passing list of categories

print(dummies)
H_D.tail(4) # This is after feature encoding
X_orig = H_D.drop("target", axis = 1)

y = H_D.iloc[:, -1].values

X_orig, y
# Normalizing data , hereX = x normalized

x = ( X_orig - np.min(X_orig) ) / ( np.max(X_orig) -  np.min(X_orig) ).values
from sklearn.model_selection import train_test_split
# I have used iloc and pandas df method here

X_orig = H_D.drop("target", axis = 1)

y = H_D.iloc[:, -1].values

X_orig, y
x = (X_orig - np.min(X_orig)) / (np.max(X_orig) - np.min(X_orig)).values
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#Transposing 

X_train = X_train.T

y_train = y_train.T

X_test = X_test.T

y_test = y_test.T
#initialize

def initialize(dimension):



    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b
def sigmoid(z):

    

    A = 1/(1+ np.exp(-z))

#     return (1/(1+ np.exp(-z)))

    return A



import numpy as np

sigmoid(np.array([0,2]))
m = X_train.shape[1]

m
def forwardBackward(w,b,X_train,y_train):

    # Forward

    

    Z = np.dot(w.T,X_train) + b

    A = sigmoid(Z)

    loss = -(y_train*np.log(A) + (1-y_train)*np.log(1- A))

    cost = np.sum(loss) / m

    

    # Backward

    dw =  np.dot(X_train,((A-y_train).T))/ m

    db =  np.sum(A - y_train)/ m

    gradients = {"dw" : dw, "db" : db}

    

    return cost,gradients

def update(w,b,X_train,y_train,learningRate,iteration) :

    costList = []

    index = []

    

    #for each iteration, update weight and bias values

    for i in range(iteration):

        cost,gradients = forwardBackward(w,b, X_train,y_train)

        w = w - learningRate * gradients["dw"]

        b = b - learningRate * gradients["db"]

        

        costList.append(cost)

        index.append(i)

    

    parameters = {"weight": w,"bias": b}

    

    print("iteration:",iteration)

    print("cost:",cost)



    plt.plot(index,costList)

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()



    return parameters, gradients
def predict(w,b, X_test):

    # After update again calculating Z and A(with updated w & b)

    Z = np.dot(w.T,X_test) + b

    A = sigmoid(Z)



    y_prediction = np.zeros((1,X_test.shape[1]))

    

    for i in range(A.shape[1]):

        if A[0,i] >= 0.5:

            y_prediction[0,i] = 1

        else:

            y_prediction[0,i] = 0

    return y_prediction

def logistic_regression(X_train,y_train,X_test,y_test,learningRate,iteration):

    

    dimension = X_train.shape[0]

    w,b = initialize(dimension)

    

    parameters, gradients = update(w,b,X_train,y_train,learningRate,iteration)



    y_prediction =  predict(parameters["weight"],parameters["bias"],X_test)

    

    print("Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
logistic_regression(X_train,y_train,X_test,y_test,0.05,1000)