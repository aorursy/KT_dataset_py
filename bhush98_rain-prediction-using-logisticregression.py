#Importing our libraries that we need



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt
# Reading our rain data using the pandas function read_csv()



df = pd.read_csv("../input/weatherAUS.csv")

df.head()
# Dropping the columns that are not necessary



df = df.drop(columns = ['Evaporation','Sunshine'])
#After dropping the columns returning first five instances of the data 



df.head()
# Dropping the NaN values from the data as they can be problematic 

# the dropna function of pandas removes the entire row in the Nan is present in any of the column



df.dropna(inplace=True)
# Returning the data to check if it removed correctly or not



df
# Using labelEncoder to assign numeric values to the string data , according to the label.



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Location'] = le.fit_transform(df['Location'])

df.head()
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])

df.head()
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])

df.head()
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])

df.head()
df['RainToday'] = le.fit_transform(df['RainToday']) 

df.head()
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

df.head()
#The describe function helps describing our data 



df.describe()
# The info function gives information about the data



df.info()
# Getting our inputs for the classifier storing it in variable X

# the .values function gets the values from the dataframe and converts it into a numpy array



X = df.iloc[:,1:-1].values

print(X[0:5,:])
#Getting the output for the classifier and storing it into variable y



Y = df['RainTomorrow'].values

print(Y[0:5])
#Importing the train_test_split function to split our data into training and testing 



from sklearn.model_selection import train_test_split

X_train,x_test,Y_train,y_test = train_test_split(X,Y)



#Printing the first five instances of the X_train



print(X_train[0:5,:])
#Printing the first five instances of the Y_train



print(Y_train[0:5])
#Getting our classifier, we are using LogisticRegression in this case.



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='lbfgs')
# Fitting our classifier to our training data



classifier.fit(X_train,Y_train)
# Getting the accuracy score on our testing data



classifier.score(x_test,y_test)
# Using the cross_val_score to divide our data into multiplt splits and check for accuracy

# One way to stop overfitting



from sklearn.model_selection import cross_val_score

results = cross_val_score(classifier,X,Y,cv=3)
# Printing the results 

# best case we are getting 98% accuracy score , it's great



print(results)