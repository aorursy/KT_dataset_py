import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualizing

import copy
dataset = pd.read_csv("../input/admission-prediction/Admission_Predict.csv")

pd.set_option('display.max_columns',len(dataset.columns))

dataset.head(5)# gives us the first five of the sample dataset
#Let's look at the exact columns' name

dataset.columns
# we are goning to seprate the class label ( Chance of Admit ) from the rest.

Y = dataset.iloc[:,-1].values

X = dataset.drop(["Serial No.","Chance of Admit "],axis=1)
dataset.info()
X.describe()
columns = X.columns

fig = plt.figure(figsize=(8,8))

for i in range(0,7):



    ax = plt.subplot(3, 3, i+1)

    ax.hist(X[columns[i]],bins = 20, color = 'blue', edgecolor = 'black')

    

    #set title name of each

    ax.set_title(columns[i])

    

plt.tight_layout()

plt.show()
X.corr()
import seaborn as sb # for vuisualizing

fig, ax = plt.subplots(figsize=(7,7))

sb.heatmap(X.corr(),linewidth = 0.5,annot=True)
from sklearn.model_selection import train_test_split #spiliting

X_train,X_test,Y_train,Y_test = train_test_split(X.values,Y,test_size = 0.25,random_state = 1)
from sklearn.linear_model import LogisticRegression # Logistic Regression to predict

classifier = LogisticRegression(random_state =0)

classifier.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score # estiating the model

from sklearn.linear_model import LinearRegression # use instead of logistic reg. to get probablistic output then use threshold to scale.



# make an indivisual logistic_regression

def Logistic_Regression(X_train,X_test,Y_train,Y_test,threshold = 0.5):

    #fitting our model for current dataset

    regressor = LinearRegression()

    regressor.fit(X_train,Y_train)

    #predict

    Y_pred = regressor.predict(X_test)

    

    Y_test_temp = copy.deepcopy(Y_test)

    for index in range(0,len(Y_pred)):

        if Y_pred[index] >= threshold:

            Y_pred[index] = 1

        else:

            Y_pred[index] = 0

    for index2 in range(0,len(Y_test)):

        if Y_test[index2] >= threshold:

            Y_test_temp[index2] = 1

        else:

            Y_test_temp[index2] = 0

    return Y_test_temp,Y_pred



# our threshold list

threshold_list = [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]

for i in threshold_list:

    Y_test_temp,Y_pred = Logistic_Regression(X_train,X_test,Y_train,Y_test,threshold = i)

  

    #now we can get the accuracy for current model

    print("ACCURACY OF ",i," THRESHOLD : ",accuracy_score(Y_test_temp,Y_pred),'\n')      
Y_test,Y_pred = Logistic_Regression(X_train,X_test,Y_train,Y_test,threshold = 0.75)

print("ACCURACY OF ",0.75 ," THRESHOLD : ",accuracy_score(Y_test,Y_pred),'\n')
# now lets see the confusion matrix of it

cm = confusion_matrix(Y_test,Y_pred)

print(cm)