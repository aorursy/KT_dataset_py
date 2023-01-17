"""

Here I am going to build my own model to predict the total number of people that survived in the famous Titanic disaster.

All data used to run this are gotten from Kaggle Titanic competition

I used Logistic regression for my model

"""
#Let's start by importing basic Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#importing data

TrainSet = pd.read_csv('train.csv')

TestSet = pd.read_csv('test.csv')
#Some little checks to determine how many missing data we have through the columns

for col in TrainSet.columns.tolist():

        print('{} has a toltal of {} missing data'.format(col, TrainSet[col].isnull().sum()))

        print('\n')

#Also let us view our Dataset to see what is needed and what is not in calculating the survival of the passengers

TrainSet.head()
#Let us do the same thing for our TestSet

TestSet.head()
"""

Here are a few things we observed:

1. Our training and test data have some missing vairables

2. Not every column is important to determining the survival of the pasanger (at least not for this model).

    Columns such as:

    * Name

    * Ticket

    * Fare (Not important as Pclass already covers this)

    * Cabin

    * Embarked

3. Our given test set does not have the dependent variable (Survived column)

Based on these observations, we will:

- replace the missing values with the median of the column

- remove the not too necessary columns

- take out the dependent variable (survived column) from the training set



"""
#Taking out the dependent variable from the training set

Y_train = TrainSet.iloc[:, 1].values

X_train = TrainSet.iloc[:,[0,2,4,5,6,7]].values
#Missing data from our Training data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)

imputer = imputer.fit(X_train[:, 3:4])

X_train[:, 3:4] = imputer.transform(X_train[:, 3:4])

#Let us split the training set X_train to create a temporary test set from the training set data.

#This is to help us review our model as dependent column was not given in the original test set

from sklearn.cross_validation import train_test_split

X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = train_test_split(X_train, Y_train, test_size = 0.33, random_state = 0)

# Do not worry about any warning sign when you run the above. It is just telling us that cross_validation will soon be removed

# as a python class and replaced with model_selection so worry not
#Now let's see what our new temp set of data look like

print(X_train_temp[0:10, 0:8])

print('\n')

print(X_test_temp[0:10, 0:8])
#Encoding categorical data 'Sex' from the temporary data:

#Temporary Training Set

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X_train_temp[:, 2] = labelencoder_X.fit_transform(X_train_temp[:, 2])

onehotencoder_X = OneHotEncoder(categorical_features = [2])

X_train_temp = onehotencoder_X.fit_transform(X_train_temp).toarray()



#Temporary Test Set

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_test_temp[:, 2] = labelencoder_X.fit_transform(X_test_temp[:, 2])

onehotencoder_X = OneHotEncoder(categorical_features = [2])

X_test_temp = onehotencoder_X.fit_transform(X_test_temp).toarray()

#Feature Scaling the Train and Test Set

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_temp = scaler.fit_transform(X_train_temp)

X_test_temp = scaler.transform(X_test_temp)

#Now Let's train with our Temp Training Set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train_temp, Y_train_temp)
#Using our trained model to predict the temporary dependent variable

Y_pred_temp = classifier.predict(X_test_temp)
# Now let us use Matrix of confusion to see how accurate our prediction was

from sklearn.metrics import confusion_matrix

cm_temp = confusion_matrix(Y_test_temp, Y_pred_temp)

print(cm_temp)
# From our confution matrix above, we can see that we predicted 152 No (Did not survive) correctly

# and predicted 80 yes correctly. This brings us to a total of 232 correctly predicted out of 295. 

# This means our prediction for the temporary set of Data had 78.64% accuracy

# Now we will visualize this more in a graph

# Now we will combine our training set back to predict the test set. This should have more accuracy as we will be having more data to train
""" Remember we have these below already

Y_train = TrainSet.iloc[:, 1].values

X_train = TrainSet.iloc[:,[0,2,4,5,6,7]].values

"""

#Also, remember we created our temporary data from the training data.

#Now we will also include the original test set

X_test = TestSet.iloc[:,[0,1,3,4,5,6]].values
# Let's take care of the Missing data from the Test data

imputerT = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)

imputerT = imputerT.fit(X_test[:, 3:4])

X_test[:, 3:4] = imputerT.transform(X_test[:, 3:4])



print(X_test[:10, 0:])
# Now let us take care of the categorical data in both train and test set. 

#Remember we have already imported LabelEncoder and OneHotEncoder created their respective objects

#Training Set

X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])

X_train = onehotencoder_X.fit_transform(X_train).toarray()



#Test Set

X_test[:, 2] = labelencoder_X.fit_transform(X_test[:, 2])

X_test = onehotencoder_X.fit_transform(X_test).toarray()

#Feature Scaling the Train and Test Set

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Now Let's train with our main Training Set and predict the test sets

classifier.fit(X_train, Y_train)



Y_pred = classifier.predict(X_test)

# Let us see what our prediction looks like. Note that we dont have the actual values for the test set to calculate confusion matrix

# So we use a simple function to check the total number of 'YES' and 'NO' our model predicted



def summary(predicted):

    predictList = predicted.tolist()

    sum_no = 0

    sum_yes = 0

    for i in range(len(predictList)):

        if predictList[i] == 0:

            sum_no = sum_no + 1

        else:

            sum_yes = sum_yes + 1

    print("Total Predicted \"NO\" (Did Not Survive) = {} \nTotal predicted \"YES\" (Survived) = {}.".format(sum_no, sum_yes))



summary(Y_pred)
