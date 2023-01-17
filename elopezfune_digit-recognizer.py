import numpy as np   #Importing Numpy

import pandas as pd  #Importing Pandas



#Data visualization

import matplotlib    #Importing Matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

matplotlib.rc('font', size=16)                #Use big fonts and big plots

plt.rcParams['figure.figsize'] = (10.0,10.0)    

matplotlib.rc('figure', facecolor='white')



import seaborn as sns #Importing Seaborn
digit_dataframe = pd.read_csv('../input/digit-recognizer/train.csv') #Importing the database

digit_dataframe.head(20) #Visualize the first 5 rows and the colunms of the database
# Database cleaning and preparation for the analysis

from sklearn.model_selection import train_test_split



#Setting up the training and test variables. The test size is 20% of the total amount of data.

X = digit_dataframe.iloc[:,1:].values

Y = digit_dataframe.iloc[:,0].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.3)
Y[0:20]
_, axarr = plt.subplots(20,20,figsize=(10,10))

for i in range(20):

    for j in range(20):

        axarr[i,j].imshow(X[int(np.linspace(10*i+j,10*i+j+1,1))].reshape((28,28), order = 'F'))          

        axarr[i,j].axis('off') 
for el in range(10):

    print("There are",len(digit_dataframe[digit_dataframe["label"]==el]),"images labeled as",el)
from sklearn.neighbors import KNeighborsClassifier



# Evaluating the accuracy of the performed analysis

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



### We will repeat the same analysis but now with the KNN algorithm



kNN = 3 #I will use this parameter and we see that the accuracy is better than the Logistic regression.

print('The k parameter is:', kNN)

classifier = KNeighborsClassifier(n_neighbors = kNN, p = 2, metric = 'euclidean' )

classifier.fit(X_train, Y_train)

Y_pred_knn = classifier.predict(X_test)



c_matrix_knn = confusion_matrix(Y_test, Y_pred_knn)

accgoaal_knn = accuracy_score(Y_test, Y_pred_knn)

print(c_matrix_knn)

print("The accuracy goal is:",round(accgoaal_knn*100,2),"%")





#sns.heatmap(c_matrix_knn, annot=True, square=True, cmap = 'Reds_r')

#plt.xlabel('Predicted')

#plt.ylabel('Actual')
digit_dataframe_test = pd.read_csv('../input/digit-recognizer/test.csv') #Importing the database

digit_dataframe_test.head() #Visualize the first 5 rows and the colunms of the database



X_to_test = digit_dataframe_test.iloc[:,:].values



_, axarr_to_test = plt.subplots(20,20,figsize=(10,10))

for i in range(20):

    for j in range(20):

        axarr_to_test[i,j].imshow(X_to_test[int(np.linspace(10*i+j,10*i+j+1,1))].reshape((28,28), order = 'F'))          

        axarr_to_test[i,j].axis('off') 

        

Y_prediction_knn = classifier.predict(X_to_test)

Y_prediction_knn[0:100]



submissions_knn=pd.DataFrame({"ImageId": list(range(1,len(Y_prediction_knn)+1)),

                         "Label": Y_prediction_knn})

submissions_knn.to_csv("sample_submission_knn.csv", index=False, header=True)