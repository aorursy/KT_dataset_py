import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#training data
train_set=pd.read_csv("../input/train.csv")
Y_train=train_set.iloc[:, 1].values #dependent: Survived
X_train=train_set.iloc[:, [2,4,5]].values #independent: Gender, Age, Class

#test data
test_set=pd.read_csv("../input/test.csv")
#Y_pred= Y_test
X_test=test_set.iloc[:, [1,3,4]].values #independent: Gender, Age, Class


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1]) # encodes vlaues of first column
print(X_train)

labelencoder_X2 = LabelEncoder()
X_test[:, 1] = labelencoder_X2.fit_transform(X_test[:, 1]) # encodes vlaues of first column

print(X_test)

#Taking care of missing values
from sklearn.preprocessing import Imputer #scikit learn : libraries for ML models
                                          #Imputer class: take care of missing data
                                          
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #recognises missing values
                                                             #mean strategy
imputer = imputer.fit(X_train[:, :])

X_train[:, :] = imputer.transform(X_train[:, :])
print(X_train)

imputer2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #recognises missing values
                                                             #mean strategy
imputer2 = imputer.fit(X_test[:, :])

X_test[:, :] = imputer.transform(X_test[:, :])
print(X_test)
#Visualising the Training set results
plt.scatter(X_train[:, 1], Y_train, color = 'red')
plt.plot(X_train[:,1], regressor.predict(X_train), color = 'blue')
plt.title('Survival vs Gender (Training set)')
plt.xlabel('Gender')
plt.ylabel('Survival')
plt.show()
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set results
Y_test = regressor.predict(X_test)
print(np.around(Y_test.reshape(418,1)))

