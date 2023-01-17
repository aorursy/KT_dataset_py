#Importing visualizing libraries like pyplot and seaborn

import matplotlib.pyplot as plt

import seaborn as sns



# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 



#For accessing data

import os



#Setting style of seaborn plots. You can choose this based on your personal preference.

sns.set_style('darkgrid')



#This command will plot the pyplot graphs below the cell itself 

%matplotlib inline



#ML libraries for train and test data split, and ML model like Linear Regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



#Data will be fed to the model in the form of a numpy array

import numpy as np
student_data = pd.read_csv("../input/student-por.csv")

student_data.head()
student_data = pd.read_csv('../input/student-por.csv', sep=";")

student_data.head()
#Let's look at some stats of our dataset

student_data.describe()
#Let's look at the column names, shall we?

list(student_data.columns)
#Selecting the above mentioned columns as our parameters

data = student_data[['studytime', 'failures', 'paid', 'absences', 'G1', 'G2', 'G3']]

data.head()
'''

Since some of the values are stored as 'yes' or 'no' in the dataframe, we need to convert 

them to 1 or 0 before training on them. This function will accept a dataframe and

the column name which we wish to convert. Please note that the conversion will be done in place.

'''

def yes_no_converter(df, col):

    df[col].replace('yes', 1, inplace=True)

    df[col].replace('no', 0, inplace=True)
yes_no_converter(data, 'paid')

data.head()
plt.figure()

plt.title("Grades obtained vs studytime")

sns.swarmplot(data=data, x='studytime', y='G3')

plt.figure()

sns.swarmplot(data=data, x='failures', y='G3')

plt.title("Failures vs Scores obtained in G3")

plt.figure()

sns.scatterplot(data=data, x='absences', y='G3')

plt.title("Absences vs Scores obtained in G3")
plt.figure()

sns.scatterplot(data=data, x="G1", y='G3')

plt.title("Co-relation between G1 and G3")

plt.figure()

sns.scatterplot(data=data,x="G2", y='G3')

plt.title("Co-relation between G2 and G3")
#Since we're gonna predict G3, we'll remove it from our parameters.

predict = "G3"

X = np.array(data.drop([predict], 1))

Y = np.array(data[predict])

#Splitting the data into training and test sets

X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, test_size=0.1)
#Defining and fitting a linear regression model

model = LinearRegression()

model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)

print(score)
predictions = model.predict(X_test)

print('Predicted Grade\t\tActual Grade')

for i in range(10):

    print('{:.2f}\t\t\t{}'.format(predictions[i],Y_test[i] ))