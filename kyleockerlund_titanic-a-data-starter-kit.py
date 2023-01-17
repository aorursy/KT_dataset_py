import numpy as np # Numpy handles all things Linear Algebra
import pandas as pd  # Pandas handles DataFrames, the things that we use to structure our data

#Then, to help visualize our data, we use matplotlib and seaborn. 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.describe()
data_train.sample(5)
data_train["Age"].plot(kind="hist", title="Ages of Titanic Passengers")
data_train.groupby("Survived")["Age"].plot(kind="hist", title="Ages of Titanic Survivers vs. Casulities", legend=True)
data_train.groupby("Survived").describe()
#This is a function. It takes in a DataFrame X, cleans it up, and returns the result. We can reuse this wherever we may need, whenever we may need. 

def cleanInput(x):
    #In this case, passenger ID, the ticket number, and the cabin number seem to be fairly bad indicators. Let's just drop them all together. 
    x = x.drop("PassengerId", axis=1) #axis=1 tells the command to drop a column, not a row.
    x = x.drop("Ticket", axis=1) 
    x = x.drop("Cabin", axis=1) 
    x = x.drop("Name", axis=1)
    
    #One-hot encode the given variable
    x = pd.get_dummies(x)
    
    #Lastly, replace all 0's. 
    x = x.fillna(0)
    return x
#Now that we declared our function above, we can pass in the training data to it, get the result, and finally that result.
#Best of all, because that code is in a function, we can call it over and over again.
cleanInput(data_train).head()
#Seperate our x variables from our y variables
from sklearn.model_selection import train_test_split

x_all = cleanInput(data_train.drop(['Survived'], axis=1))
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=23)
X_train.sample(4)
y_train.sample(4)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

model = RandomForestClassifier()
model = LinearSVC() #SVC = Support Vector Classifier
#You could use the above to keep subbing data in and out. Or, you could use the below to 
model = KNeighborsClassifier(n_neighbors = 4)

# Fit the best algorithm to the data. 
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
#Predict on the final test variables
ids = data_test["PassengerId"]
predictions = model.predict(cleanInput(data_test))
submission = pd.DataFrame({"PassengerId": ids,"Survived": predictions})
#Ok, everything seems to be in order... Note: The competition runners will specify the format. In this case, they provided a sample submission
#here (it's the gendered_submission. For the sake of demonstration, they just said every female survived, every male didn't.
#https://www.kaggle.com/c/titanic/data)
print(submission.sample(10))
submission.to_csv('titanic.csv', index=False)
