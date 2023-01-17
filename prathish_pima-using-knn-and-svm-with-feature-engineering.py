# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Code by PRATHISH  MURUGAN

# 29 April 2020



#Pima-Indians-Diabetes



#This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.  

#he datasets consist of several medical predictor (independent) variables and 

#one target (dependent) variable, Outcome. 

#Independent variables include the number of pregnancies the patient has had, their BMI, 

#insulin level, age, and so on.



#This dataset can be found here -> https://www.kaggle.com/uciml/pima-indians-diabetes-database



#Acknowledgements :-

#Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 

#Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. 

#In Proceedings of the Symposium on Computer Applications and 

#Medical Care (pp. 261--265).IEEE Computer Society Press.



#This is a classifiaction problem and the dataset is not clean



#This problem looks easy bit it gets hard because of the unclean and unclean dataset
import matplotlib.pyplot as plt         #graphs

import seaborn as sns                   #visualizations


#Imporing the CSV dataset

pima_main=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')



#Making a copy of the original dataset for working

pima_work=pima_main.copy(deep=True)



print(pima_work.head())

print(pima_work.shape)


pima_work.describe()   # Gives the basic mean,meadian and quantas og each columns

#Note atleast 25% of readings for insulin and skinthickness are 0. Min readings 

#for columns like Glucose, Bloodpressure and BMI are zero too which does not seem appropriate



pima_main.info()

pima_work.isnull().sum()     #to check any null values in any columns

#There are no null values in the dataset
#Now we have to check if there are any 0's in the set

pima_work.isin([0]).sum()

# Seems like there are a lot of zeros and 0's in glucose,BP,BMI is not logical

#We need to fill these 0's with some appropirate values



pima_work.Outcome.value_counts()

#Seems like the Outcome is not Balanced
# replace all zeros with NANs. I do this so that when means are calculated, zeros are not counted.

pima_nan = pima_work.replace({

            'Glucose': 0,

            'BloodPressure' : 0,

            'SkinThickness' : 0,

            'BMI' : 0 ,

            'Insulin' : 0,

        },np.NaN)                     





pima_nan.isin([0]).sum()
#We have replaced all the unwanted 0's with NaN 

#Calcutating the mean of each columns

pima_nan.mean()

pima_nan.median()

#Now we will try to replace this NaN with mean or median

pima_nan.isnull().sum()

pima_nan=pima_nan.fillna(pima_nan.mean())

pima_nan.isnull().sum()

#Now we have filled the NaN values with the mean of each coloums
#shuffle the dataset

from sklearn.utils import shuffle

pima_nan = shuffle(pima_nan)



# I want to see how the imputation process has affected these values. 

# I will follow this up with some visualizations.



pima_nan.groupby('Outcome').mean().transpose()
pima_corr=pima_nan.corr()



#Creating a Heatmap

plt.figure(figsize=(9,9))

sns.heatmap(pima_corr,cmap='coolwarm',annot=True,linewidths = 0.5)



#From this heatmap we can see that Glucose is having the greatest

#effect on outcome followed by BMI which is self-explanatory.
# Histograms for imputed data.



for cols in pima_nan.columns:

    x = pima_nan.loc[:,cols]

    plt.hist(x)

    plt.title('Histogram for Feature' +str(cols))

    plt.show()

# End of Visualizations
#Creating a test and train DF    

X=pima_nan.drop('Outcome',axis=1)

y=pima_nan['Outcome'].values


#Let's split the data randomly into training and test set

    #importing train_test_split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=42)
#ML MODELS



# MODEL 1 :- KNN CLASSIFICATION



#Create-KNN-model

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 60)       #n_neighbors = K value



#K-fold CV

from sklearn.model_selection import cross_val_score

accuraccies = cross_val_score(estimator = KNN, X= X_train, y=y_train, cv=10)

print("Average Accuracies: ",np.mean(accuraccies))

print("Standart Deviation Accuracies: ",np.std(accuraccies))



#The values I got for KNN at the time of coding is 

#Average Accuracies:  0.7664285714285715

#Standard Deviation Accuracies:  0.09502140527437733
KNN.fit(X_train,y_train) #learning model

KNN.score(X_test, y_test)



#The score I got while writing the score 

#Score :-  0.751219512195122
y_predict = KNN.predict(X_test)

from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test,y_predict)



#CM visualization



import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Prediction(Ypred)")

plt.ylabel("Ytest")

plt.show()
# MODEL 2 :- SVM



#Create SVM Model

from sklearn.svm import SVC



SVM = SVC(random_state=42)



#K-fold CV

from sklearn.model_selection import cross_val_score

accuraccies = cross_val_score(estimator = SVM, X= X_train, y=y_train, cv=10)

print("Average Accuracies: ",np.mean(accuraccies))

print("Standard Deviation Accuracies: ",np.std(accuraccies))



#The values I got for SVM at the time of coding is 

#Average Accuracies:  0.6410714285714286

#Standard Deviation Accuracies:  0.018064274887492272
SVM.fit(X_train,y_train)  #learning 

#SVM Test 

print ("SVM Accuracy:", SVM.score(X_test,y_test))



SVMscore = SVM.score(X_test,y_test)

# Score :-  0.6536585365853659
#Confusion Matrix



yprediciton3= SVM.predict(X_test)

ytrue = y_test



from sklearn.metrics import confusion_matrix

CM = confusion_matrix(ytrue,yprediciton3)



#CM visualization



import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Prediction(Ypred)")

plt.ylabel("Ytrue")

plt.show()
# So this is a machine learning model to accurately predict whether or not the

# patients in the dataset have diabetes or not



# Hope you find it useful

# Corections and suggestions are welcomed



# BY PRATHISH MURUGAN

# 29 - APRIL - 2020