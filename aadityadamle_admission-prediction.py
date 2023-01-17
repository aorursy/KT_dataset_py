# import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# loading the data from csv file saved at the url
#data = pd.read_csv("https://raw.githubusercontent.com/abcom-mltutorials/Admit-Prediction/master/Admission_Predict_Ver1.1.csv")
data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
# observing the data with the first 5 rows 
data.head()
# finding the no. of rows and columns in the dataset
data.shape
# checking null items
print(data.isna().sum())
# We use drop function to delete the column
data = data.drop(["Serial No."], axis = 1)
# Extractig features
X = data.iloc[:,:7]
X.info()
# Extracting target
y = data.iloc[:,7:]
y.info()
# Split data into tarining and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X, y, 
                                                 random_state = 10, 
                                                 shuffle = True, 
                                                 test_size = 0.2)
# Visualize the effect of GRE Score on chance of getting an admit  
plt.scatter(X_train["GRE Score"],Y_train, color = "red")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admission")
plt.legend(["GRE Score"])
plt.show()
# Visualize the effect of CGPA on chance of of getting an admit. 
plt.scatter(X_train["CGPA"],Y_train, color = "green")
plt.xlabel("CGPA")
plt.ylabel("Chance of Admission")
plt.legend(["CGPA"])
plt.show()
# Visulize the University Rating vs. Chance of Admission.   
plt.scatter(X_train["University Rating"],Y_train, color = "blue")
plt.xlabel("University Rating")
plt.ylabel("Chance of Admission")
plt.legend(["University Rating"])
plt.show()
# Loading the classifier from the sklearn
classifier = LinearRegression()
# Fitting the classifier onto the training data
classifier.fit(X_train,Y_train)
#Using the classifier to predict the test data and storing it in prediction_of_y array
prediction_of_Y = classifier.predict(X_test)
# Printing the first six predicted values
prediction_of_Y = np.round(prediction_of_Y, decimals = 3)
prediction_of_Y[:6]
# Comparision of predicted data(prediction_of_Y) and actual data(Y_test)
Y_test["Predicted chance of Admit"] = prediction_of_Y.tolist()
print(Y_test)
# Dropping the added column to keep the dimensions of Y_test intact
Y_test = Y_test.drop(["Predicted chance of Admit"], axis = 1)
# Visualize the difference in graph for same parameter "GRE Score" for actual chance & prediction chance. 
plt.scatter(X_test["GRE Score"],Y_test, color = "red")
plt.scatter(X_test["GRE Score"], prediction_of_Y, color='purple')
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admission")
plt.legend(["Actual chance for GRE Score","Predicted chance for GRE Score"])
plt.show()
# Visualize the difference in graph for same parameter "SOP" for actual chance & prediction chance.
plt.scatter(X_test["SOP"],Y_test, color = "blue")
plt.scatter(X_test["SOP"], prediction_of_Y, color='orange')
plt.xlabel("SOP")
plt.ylabel("Chance of Admission")
plt.legend(["Actual chance for SOP","Predicted chance for SOP"])
plt.show()
print('Accuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))
# User - specified predictions. Adding my data to X_test dataset:
my_data = X_test.append(pd.Series([332, 107, 5, 4.5, 4.0, 9.34, 0], index = X_test.columns), ignore_index = True)

# Checking the dataframe for our values:
print(my_data[-1:])
#Checking my chances of admission. # As the new record is saved at the end array the result will be the last value. 
my_chance = classifier.predict(my_data)
my_chance[-1]
# Evaluating admit chance for multiple user-defined records:
list_of_records = [pd.Series([309, 90, 4, 4, 3.5, 7.14, 0], index = X_test.columns),
                   pd.Series([300, 99, 3, 3.5, 3.5, 8.09, 0], index = X_test.columns),
                   pd.Series([304, 108, 4, 4, 3.5, 7.91, 0], index = X_test.columns),
                   pd.Series([295, 113, 5, 4.5, 4, 8.76, 1], index = X_test.columns)]
user_defined = X_test.append(list_of_records, ignore_index= True)
print(user_defined[-4:]) 

chances = classifier.predict(user_defined)
chances[-4:]
#Checking chances of single record without appending to previous record
single_record_values = {"GRE Score" : [327], "TOEFL Score" : [95], "University Rating" : [4.0], "SOP": [3.5], "LOR" : [4.0], "CGPA": [7.96], "Research": [1]}
single_rec_df = pd.DataFrame(single_record_values, columns = ["GRE Score",  "TOEFL Score",  "University Rating",  "SOP",  "LOR",   "CGPA",  "Research"])
print(single_rec_df)

single_chance = classifier.predict(single_rec_df)
single_chance 