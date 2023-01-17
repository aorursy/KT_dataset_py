# This is an example on how to use LinearRegression

import warnings

warnings.filterwarnings("ignore")



# Load the diabetes dataset

import pandas as pd

boston = pd.read_csv('../input/boston-dataset/boston_dataset.csv')

y = boston['MEDV']

X = boston.drop('MEDV',axis=1)

#View the top 5 rows

boston.head()
'''

Split train and test set

'''



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.1, random_state=0)



# View the shape (structure) of the data

print(f"Training features shape: {X_train.shape}")

print(f"Testing features shape: {X_test.shape}")

print(f"Training label shape: {y_train.shape}")

print(f"Testing label  shape: {y_test.shape}")
#Linear Regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

lr = LinearRegression(normalize=True)   
lr.fit(X_train, y_train) # Fit the model to the data
y_pred_lr = lr.predict(X_test)  # Predict labels
# The mean squared error

print(f"Mean squared error: { mean_squared_error(y_test, y_pred_lr)}")

# Explained variance score: 1 is perfect prediction

print(f"Variance score: {r2_score(y_test, y_pred_lr)}")

# Mean Absolute Error

print(f"Mean squared error: { mean_absolute_error(y_test, y_pred_lr)}")