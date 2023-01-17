import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import math

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import normalize

import keras 

from keras.models import Sequential # For defining the type of the Neural Network

from keras.layers import Dense  # For defining the layers of the Neural Network

from keras.layers import Dropout # For Dropout Regularization

from keras.optimizers import RMSprop # For the RMSprop optimizer 

from keras.callbacks import ReduceLROnPlateau # For Simulated Annealing

from keras.wrappers.scikit_learn import KerasClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combined = train.append(test, ignore_index=True, sort=False)
train.info()
test.info()
combined.info()
#The columns with missing values are as follows:

# 1.Age

# 2.Fare

# 3.Cabin

# 4.Embarked
combined.head()
# The following attributes will be dropped based on a cursory look.

# PassengerId as it is not relevant for the survival of a passenger.

# Name as it is not relevant for the survival of a passenger.

# Ticket will be dropped as the passenger class and the Fare attribute makes the Ticket attribute redundant.

# Cabin attribute will be dropped due to a very high number of missing values (77 percent)

# We combine the training and test sets for simplicity in processing the data.

combined = combined.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1)
combined.head()
#Creating the training set

train1 = combined.iloc[:,1:]

train1.head()
#Now to determine the importance of each feature.

#For Pclass

#Checking the count of every value to check for bias.

pd.value_counts(train['Pclass'])
sns.lmplot(x='Pclass', y='Survived', data=train)
#We can see that Pclass plays an important role as the Pclass increases, the probability of survival decreases.
#Now, let's check Gender

pd.value_counts(train['Sex'])
#Male values are 50 percent more.

#Let's check the probability of survival of each class.

sns.catplot(x="Sex",y="Survived",kind='bar',data=train)
#The probability of female survival is more. The sex attribute therefore, will play an important role in this problem.
#Now let's look at the age attribute.

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Green", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
#We can see that increased number of babies survived as compared to other ages.
#Let's take at the number of siblings

pd.value_counts(train['SibSp'])
#We can see a bias here. Almost 70 percent of the training data has 0 SibSp value.

#Let's visualize a bar graph to check the importance of the SibSp attribute.

g  = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar")
#So as the number of siblings increases, the probability of survival decreases.

#However, due to increased bias, the importance of SibSp does not come close to the previous attributes.

#Now, let's have a look at the number of parent/child attribute.

pd.value_counts(train['Parch'])
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar")
#The probability of survival decreases as the family members increases.

#However, the deviation for the 3 parch value is pretty high.

#The importance of this attribute is a bit less than the sibling attribute.
#Now for fare.

pd.value_counts(train['Fare'])

g = sns.distplot(train['Fare'])

train['Fare'].skew()
# We can see that the data is negatively skewed. Therefore, we will apply the logarithmic transformation.
# Now, let's check for the embarked attribute

pd.value_counts(train['Embarked'])
# Pretty high bias for Embarked attribute.
g  = sns.factorplot(x="Embarked",y="Survived",data=train,kind="bar")
# The passengers from Cherbourg have a higher survival rate.

# Maybe more number of passengers from there have a  first class ticket.

# Embarked attribute might be more important than parents and sbilings but less than the first three attributes.
combined.info()
# For Age

# We will try to use Regression to predict the missing Age values.

# To determine the variables to be used in predicting the Age, we will plot a heatmap.

# We need to encode the Sex attribute before using the heatmap.

labelencoder = LabelEncoder()

combined.iloc[:, 2] = labelencoder.fit_transform(combined.iloc[:, 2])
combined.head()
g = sns.heatmap(combined[["Age","SibSp","Parch","Sex","Pclass"]].corr(),annot=True, fmt = ".2f")
# We can see that SibSp, Pclass have higher magnitudes of co-relation with Age.

# Let's try to predict the missing Age values by using SibSp and Pclass values
combined.head()
Age_pred = combined.iloc[:,1:5]

Age_pred = Age_pred.drop(["Sex"],axis = 1)

Age_pred.head()
# Our model will train on the non-null rows of Age attribute and predict the null rows of the same.

# However, for testing the model, we split the training set initially. 
# Creating the training set and the test set.

index = []

Age_Test =  []

#Here we create the test set.

for x in range(len(Age_pred)):

    if math.isnan(Age_pred.iloc[x,1]) == True:

        index.append(x)

        Age_Test.append(Age_pred.iloc[x,:])

Age_Test = pd.DataFrame(Age_Test)

Age_Test.head()

Age_Test = Age_Test.drop(["Age"],axis = 1)
#Here we create the training set

for x in range(len(index)):

    Age_pred = Age_pred.drop(index[x])
Age_pred.info()
X = Age_pred.drop(["Age"],axis = 1)

y = Age_pred.iloc[:,1]
X.head()
y.head()
Age_X_train,Age_X_test,Age_y_train,Age_y_test = train_test_split(X,y, test_size=0.2, random_state = 0)
# I hvae used the xgboost model. I have optimized the following parameters by using a simple for loop.

# 1. n_estimators

# 2. max_depth

# 3. learning_rate
regressor = XGBRegressor()

regressor.fit(Age_X_train,Age_y_train)

y_pred = regressor.predict(Age_X_test)

rms = math.sqrt(mean_squared_error(Age_y_test, y_pred))

rms
rms = []

for x in np.arange(0,0.1,0.01):

    regressor = XGBRegressor(learning_rate = x, max_depth = 1)

    regressor.fit(Age_X_train,Age_y_train)

    y_pred = regressor.predict(Age_X_test)

    rms.append(math.sqrt(mean_squared_error(Age_y_test, y_pred)))

    
regressor = XGBRegressor(max_depth = 1, learning_rate = 0.03)

regressor.fit(Age_X_train,Age_y_train)

y_pred = regressor.predict(Age_X_test)

rms = math.sqrt(mean_squared_error(Age_y_test, y_pred))

rms
# After optimization, the rms value has decreased from 12.5 to 12.1.
# Now let's try other models.

#SVR

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(Age_X_train,Age_y_train)

y_pred = regressor.predict(Age_X_test)

rms = math.sqrt(mean_squared_error(Age_y_test, y_pred))

rms
#Random Forest

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 5000, random_state = 0)

regressor.fit(Age_X_train,Age_y_train)

y_pred = regressor.predict(Age_X_test)

rms = math.sqrt(mean_squared_error(Age_y_test, y_pred))

rms
# XGBoost works best.

# We'll use XGBoost

# Predicting the missing values

regressor = XGBRegressor(max_depth = 1, learning_rate = 0.03)

regressor.fit(X,y)

y_pred = regressor.predict(Age_Test)
for x in range(len(y_pred)):

    y_pred[x] = int(y_pred[x])
y_pred
# Getting the column index of Age

combined.columns.get_loc("Age")
count = 0

for x in range(len(combined)):

    if math.isnan(combined.iloc[x,3]):

        combined.iloc[x,3] = y_pred[count]

        count = count+1
combined.info()
# For Parch and Fare, we will just replace the missing values by the most frequent and the average respectively due to the extremely small number of missing values
# For Fare

combined.columns.get_loc("Fare")

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(combined.iloc[:, 6:7])

combined.iloc[:, 6:7] = imputer.transform(combined.iloc[:, 6:7])

combined["Fare"].isnull().sum()
# For Embarked

combined.columns.get_loc("Embarked")

temp = combined["Embarked"].isnull()

for x in range(len(combined)):

    if temp[x] == True:

        combined.iloc[x,7] = "S"

combined["Embarked"].isnull().sum()
combined.info()
X = combined.drop(["Survived"],axis = 1)
y = combined.iloc[0:891,0]
len(y)
X.info()
X.head()
X.iloc[:, 6] = labelencoder.fit_transform(X.iloc[:, 6])
X.head()
# Now, we OneHotEncode the Sex and the Embarked attribute.
# For Sex

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()
X = pd.DataFrame(X)

X.head()
# For embarked

onehotencoder = OneHotEncoder(categorical_features = [7])

X = onehotencoder.fit_transform(X).toarray()
X = pd.DataFrame(X)
X.head()
X.iloc[:,9].describe()
X.iloc[:,9] = np.log(X.iloc[:,9] + 1) # + 1 to avoid divide by zero error.
g = sns.distplot(X.iloc[:,9])

X.iloc[:,9].skew()
# We shall now normalize the values to obtain better results.

X.iloc[:,9] = normalize(X.iloc[:,-1:], axis=0, norm='max')

g = sns.distplot(X.iloc[:,9])

X.iloc[:,9].skew()
X.head()
X.info()
# Now, we separate the combined set back to the training and the test set.
train_final = X.iloc[0:891,:]

train_final.info()
test_final = X.iloc[891:,:]

test_final.info()

# Dropping the Survived attribute from the test set

#test_final = test_final.drop(test_final.columns[0], axis = 1)

#test_final.info()
y.head()
# The dataset is ready.

# Since, this is a classification problem, I am going to use a Artificial Neural Network using the Keras Library. 
# Let us split X and y into training set and test set.

X_train, X_test, y_train, y_test = train_test_split(train_final,y, test_size=0.2, random_state = 0)
#I have used the used following formula for the number of nodes in the hidden layer.

#Nh=Ns/(α∗(Ni+No))

#Ni  = number of input neurons.

#No = number of output neurons.

#Ns = number of samples in training data set.

#α = an arbitrary scaling factor usually 2-10.

Nh = int(891/32)
classifier_arr = []

for x in range(10,100,10):

    # Initialising the ANN

    classifier = Sequential()



    # Adding the input layer and the first hidden layer

    classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

    #Adding dropout regularization to prevent overfitting.

    classifier.add(Dropout(0.01))



    # Adding the second hidden layer

    classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dropout(0.01))



    # Adding the output layer

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.add(Dropout(0.01))



    # Define the optimizer

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model

    classifier.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

    

    learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 

                                                patience=3, 

                                                verbose=1, 

                                                factor=0.5, 

                                                min_lr=0.00001)



    # Fitting the ANN to the Training set

    history = classifier.fit(X_train, y_train, batch_size = 25, epochs = 45, callbacks = [learning_rate_reduction])

    classifier_arr.append(classifier)

    # I have tried keeping the number of epochs as 1000 ,500 , 100 , 50. 

    # However, after observation I can see that after the learning rate is reduced by the annealer, the accuracy remains constant and then increases with further reduction in learning rate.

    # So, I have observed the epoch with  the final reduction in learning rate and have kept the total number of epochs uptill that  only.
# Predicting the results.

y_pred_arr = []

for x in range(len(classifier_arr)):

    y_pred = classifier_arr[x].predict(X_test)

    y_pred = (y_pred > 0.5)

    y_pred = y_pred.astype(int)

    y_pred_arr.append(y_pred)

    
# Let us see the Confusion Matrix.

# Creating the Confusion Matrix.

accuracy_arr = []

for x in range(len(classifier_arr)):

    

    confusion_mtx = confusion_matrix(y_test, y_pred_arr[x]) 

    accuracy = (confusion_mtx[0][0] + confusion_mtx[1][1]) / sum(sum(confusion_mtx))

    accuracy_arr.append(accuracy)

#Visualise the Confusion Matrix 

#sns.heatmap(confusion_mtx, annot=True, fmt='d')
accuracy_arr.index(max(accuracy_arr))
# However, the reduction of epochs gave an accuracy of 0.77511.

# So, I decided to increase the number of epochs.
Nh = int(891/32)

# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

#Adding dropout regularization to prevent overfitting.

classifier.add(Dropout(0.01))



# Adding the second hidden layer

classifier.add(Dense(units = Nh, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.01))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.add(Dropout(0.01))



# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model

classifier.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])



#In order to make the optimizer converge faster and closest to the global minimum of the loss function, I have used an annealing method of the learning rate (LR).

#The LR is the step by which the optimizer walks through the 'loss landscape'. The higher LR, the bigger are the steps and the quicker is the convergence. However the sampling is very poor with an high LR and the optimizer could probably fall into a local minima.

#Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.

#To keep the advantage of the fast computation time with a high LR, I have decreased the LR dynamically every X steps (epochs) if necessary (when accuracy is not improved).

#With the ReduceLROnPlateau function from Keras.callbacks, I have choosen to reduce the LR by half if the accuracy is not improved after 3 epochs.

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 

                                                patience=3, 

                                                verbose=1, 

                                                factor=0.5, 

                                                min_lr=0.00001)



# Fitting the ANN to the Training set

classifier.fit(train_final, y, batch_size = 25, epochs = 1610, callbacks = [learning_rate_reduction])
# By a lot of trial and error in the number of epochs, I got an accuracy of 0.79904
# Making the predictions.

y_pred = classifier.predict(test_final)

y_pred = (y_pred > 0.5)

y_pred = y_pred.astype(int)
# Making the final submission file.

final = pd.DataFrame(y_pred)

final['PassengerId'] = pd.Series(data = np.arange(892,1310), index=final.index)

final.columns = ['Survived','PassengerId']

columnsTitles=["PassengerId","Survived"]

final=final.reindex(columns=columnsTitles)



# Exporting the dataframe

final.to_csv('Predictions_ANN.csv', index = False)
y_pred_arr = []

from xgboost import XGBClassifier

for x in np.arange(0,0.1,0.01):

    classifier = XGBClassifier(n_estimators = 200, max_depth = 8, learning_rate = x)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    y_pred_arr.append(y_pred)

    
acc = []

from sklearn.metrics import accuracy_score

for x in range(len(y_pred_arr)):

     acc.append(accuracy_score(y_test, y_pred_arr[x]))
acc.index(max(acc))

max(acc)
classifier = XGBClassifier(n_estimators = 200, max_depth = 8)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy_score(y_test, y_pred)
classifier = XGBClassifier(n_estimators = 200, max_depth = 8)

classifier.fit(train_final, y)

y_pred = classifier.predict(test_final)

y_pred = y_pred.astype(int)

# XGBoost Classifier gave an accuracy of 0.76076
y_pred_arr = []

from sklearn.ensemble import RandomForestClassifier

for x in np.arange(100,1000,100):

    classifier = RandomForestClassifier(n_estimators = x, random_state = 0)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    y_pred_arr.append(y_pred)
acc = []

from sklearn.metrics import accuracy_score

for x in range(len(y_pred_arr)):

     acc.append(accuracy_score(y_test, y_pred_arr[x]))
acc.index(max(acc))

#max(acc)
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)

classifier.fit(train_final, y)

y_pred = classifier.predict(test_final)

#accuracy_score(y_test, y_pred)

y_pred = y_pred.astype(int)

# Random Forest gave an accuracy of 0.75598.
# Let's try Support Vector Classification

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(train_final, y)

y_pred = classifier.predict(test_final)

y_pred = y_pred.astype(int)

# Support Vector Classification gave an accuracy of 0.75598 too.