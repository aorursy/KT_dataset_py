##Importing the packages

#Data processing packages

import numpy as np 

import pandas as pd 



#Visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns 



#Machine Learning packages

from sklearn.svm import SVC,NuSVC

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler	

from sklearn.metrics import confusion_matrix



#Suppress warnings

import warnings

warnings.filterwarnings('ignore')
# The dataset contains the information of 7042 Customers and their churn value.

data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()
#Check the datatypes of the fields and null values. Below output shows that there are no missing values

data.info()
#These fields does not add value, hence removed

data = data.drop(['customerID'], axis = 1)
#A lambda function is a small anonymous function.

#A lambda function can take any number of arguments, but can only have one expression.

data['Churn']=data['Churn'].apply(lambda x : 1 if x=='Yes' else 0)
#Finding the Count of Customer Churn. The output shows that 1869 customers churned(left) last month

data.Churn.value_counts()
data.head()
#This function is used to convert Categorical values to Numerical values

data=pd.get_dummies(data)
data.head()
#Separating Feature and Target matrices

X = data.drop(['Churn'], axis=1)

y=data['Churn']
#Feature scaling is a method used to standardize the range of independent variables or features of data.

#Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X = scale.fit_transform(X)
# Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#Function to plot Confusion Matrix

def cm_plot(cm,Model):

    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

    classNames = ['Negative','Positive']

    plt.title('Comparison of Prediction Result for '+ Model)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames, rotation=45)

    plt.yticks(tick_marks, classNames)

    s = [['TN','FP'], ['FN', 'TP']]

    for i in range(2):

        for j in range(2):

            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

    plt.show()
#Function to Train and Test Machine Learning Model

def train_test_ml_model(X_train,y_train,X_test,Model):

    model.fit(X_train,y_train) #Train the Model

    y_pred = model.predict(X_test) #Use the Model for prediction



    # Test the Model

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test,y_pred)

    accuracy = round(100*np.trace(cm)/np.sum(cm),1)



    #Plot/Display the results

    cm_plot(cm,Model)

    print('Accuracy of the Model' ,Model, str(accuracy)+'%')
from sklearn.svm import SVC,NuSVC  #Import packages related to Model

Model = "SVC"

model=SVC() #Create the Model



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.svm import SVC,NuSVC  #Import packages related to Model

Model = "NuSVC"

model=NuSVC(nu=0.285)#Create the Model



train_test_ml_model(X_train,y_train,X_test,Model)
from xgboost import XGBClassifier  #Import packages related to Model

Model = "XGBClassifier()"

model=XGBClassifier() #Create the Model



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.neighbors import KNeighborsClassifier  #Import packages related to Model

Model = "KNeighborsClassifier"

model=KNeighborsClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.naive_bayes import GaussianNB,MultinomialNB  #Import packages related to Model

Model = "GaussianNB"

model=GaussianNB()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.linear_model import SGDClassifier, LogisticRegression #Import packages related to Model

Model = "SGDClassifier"

model=SGDClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.linear_model import SGDClassifier, LogisticRegression #Import packages related to Model

Model = "LogisticRegression"

model=LogisticRegression()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier #Import packages related to Model

Model = "DecisionTreeClassifier"

model=DecisionTreeClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier #Import packages related to Model

Model = "ExtraTreeClassifier"

model=ExtraTreeClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis #Import packages related to Model

Model = "QuadraticDiscriminantAnalysis"

model = QuadraticDiscriminantAnalysis()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis #Import packages related to Model

Model = "LinearDiscriminantAnalysis"

model=LinearDiscriminantAnalysis()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model

Model = "RandomForestClassifier"

model=RandomForestClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model

Model = "AdaBoostClassifier"

model=AdaBoostClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #Import packages related to Model

Model = "GradientBoostingClassifier"

model=GradientBoostingClassifier()



train_test_ml_model(X_train,y_train,X_test,Model)