import pandas as pd

import numpy as np

data = pd.read_csv("../input/covid-19-nlp-text-classification/Corona_NLP_test.csv")

print("First 11 rows: ")

data.head(11)
print(" \nThe row with a missing value : ")

data[data['Location'].isnull()]
print(" \nReplacing all the null with nan")

data[data.isnull().any(axis=1)].replace(0,np.nan)
# Question 2



# Importing matplotlib library



from matplotlib import pyplot as plt



team=['CSK','KKR','DC','MI']

points=[213,167,181,203]



# Plotting the graph

plt.bar(team, points, width = 0.8, color=['blue','green','green','green'])

plt.xlabel('Team')

plt.ylabel('Points')

plt.title('Team Points Chart')

plt.show()
# Question 3



# Importing numpy package

import numpy as np



# Creating two numpy arrays

a = np.array([10,20,30,40,50,60,70,80,90])

print("Array 1: ",a)



b = np.array([10,20,30,80,90])

print("Array 2: ",b,"\n")



# Displaying the common elements between the two arrays

print("Common values between two arrays:")

print(np.intersect1d(a, b),"\n")



# Deleting the common elements from array1

for i, val in enumerate(a):

    if val in b:

        a = np.delete(a, np.where(a == val)[0][0])



for i, val in enumerate(b):

    if val in a:

        a = np.delete(a, np.where(a == val)[0][0])

        

# Displaying the final two arrays        

print("Arrays after deletion of common elements : ")

print(a)

print(b)
# Question 4



import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



# Importing the iris dataset

train = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")



# Training dataset

X = train.drop("species",axis=1)

y = train["species"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



# Calculating F1 scores

print("F1 Score(macro):",f1_score(y_test, predictions, average='macro'))

print("F1 Score(micro):",f1_score(y_test, predictions, average='micro'))

print("F1 Score(weighted):",f1_score(y_test, predictions, average='weighted')) 



# Printing confusion matrix

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)
