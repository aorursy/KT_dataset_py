import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.preprocessing import StandardScaler 

data_sky= pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv",header = 0)

data_sky.head()
#Dropping the id feature

data_sky.drop(columns = ['objid'], inplace = True)

data_sky.head()



#Converting non-numeric data to numeric dataset

diag_map = {'STAR':1, 'GALAXY':2, 'QSO':3}

data_sky['class'] = data_sky['class'].map(diag_map)



#Preparing the data set

class_all = list(data_sky.shape)[0]

class_categories = list(data_sky['class'].value_counts())



print("The dataset has {} classes, {} stars, {} galaxies and {} quasars.".format(class_all, 

                                                                                 class_categories[0], 

                                                                                 class_categories[1],

                                                                                 class_categories[2]))

data_sky.describe()
#Creating training and test datasets

y = data_sky["class"].values

X = data_sky.drop(["class"], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)
#Training Model

classifier = KNeighborsClassifier(n_neighbors=4)  

classifier.fit(X_train, y_train) 
#Testing the model

y_pred = classifier.predict(X_test)  

print(np.mean(y_pred != y_test))

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  
#Improve Model Performance

#z-score transformed 

scaler = StandardScaler()  

scaler.fit(X_train)



#Training the model

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test) 

classifier = KNeighborsClassifier(n_neighbors=4)  

classifier.fit(X_train, y_train) 



#Testing the model

y_pred = classifier.predict(X_test)  

print(np.mean(y_pred != y_test))

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  
error = []



# Calculating error for K values between 1 and 300

for i in range(1, 300):  

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))

    

plt.figure(figsize=(12, 6))  

plt.plot(range(1, 300), error, color='red', linestyle='dashed', marker='o',  

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')  

plt.xlabel('K Value')  

plt.ylabel('Mean Error') 



plt.show()
#Training Model for better accuracy (change n neighbour value)

from sklearn.neighbors import KNeighborsClassifier  

classifier = KNeighborsClassifier(n_neighbors=50)  

classifier.fit(X_train, y_train)



#Testing the model

from sklearn.metrics import classification_report, confusion_matrix  

y_pred = classifier.predict(X_test)  

print(np.mean(y_pred != y_test))

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred)) 