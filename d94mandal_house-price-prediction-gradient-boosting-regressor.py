import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import random
def handle_non_numerical_data(data):
    "Converting all non-numerical data to numerical"
    #Store all of the columns into variable
    columns=data.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if(data[column].dtype!=np.int64 and data[column].dtype!=np.float64):
            #dtype is used to check the data type of the column
            column_contents=data[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if(unique not in text_digit_vals):
                    text_digit_vals[unique]=x
                    x+=1
            data[column]=list(map(convert_to_int,data[column]))
    return data
train_data=pd.read_csv("../input/train.csv")
train_data=train_data.drop("Id",1)
#We can't perform mathematical operation on string.That's why we have to handle non numerical data
train_data=handle_non_numerical_data(train_data)
# random.shuffle(train_data)
m=train_data.shape[0]
n=train_data.shape[1]
X=train_data.iloc[:,0:n-1]    
ones=np.ones([X.shape[0],1])
X=np.concatenate((ones,X),axis=1)
X=np.array(X)
Y=train_data.iloc[:,n-1:n].values
Y=np.array(Y).flatten()
split_1=int(0.6*len(X))
split_2=int(0.8*len(X))
X_train=X[:split_1]
Y_train=Y[:split_1]
X_dev=X[split_1:split_2]
Y_dev=Y[split_1:split_2]
X_test=X[split_2:]
Y_test=Y[split_2:]
plt.title("X_train Vs. Y_train")
plt.xlabel('Features of Houses')
plt.ylabel('Price')
plt.plot(X_train,Y_train,'ro')
plt.show()
model=GradientBoostingRegressor()
model=model.fit(np.nan_to_num(X_train),Y_train)
# print(model.score(np.nan_to_num(X),Y))
print(model.score(np.nan_to_num(X_dev),Y_dev))
print(model.score(np.nan_to_num(X_test),Y_test))
X_dev.shape
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model=model.fit(np.nan_to_num(X_train),Y_train)
print(model.score(np.nan_to_num(X_dev),Y_dev))
print(model.score(np.nan_to_num(X_test),Y_test))
# model.predict(np.nan_to_num(X_test))
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model=model.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train,model.predict(np.nan_to_num(X_train)))
test_data=pd.read_csv("../input/test.csv")
test_data=test_data.drop("Id",1)
test_data=handle_non_numerical_data(test_data)
X_test=test_data.iloc[:,0:n-1]
ones=np.ones([X_test.shape[0],1])
X_test=np.concatenate((ones,X_test),axis=1)
X_test=np.array(X_test)
predicted_prices=model.predict(np.nan_to_num(X_test))
print(predicted_prices)
plt.title("X_test Vs. Y_test")
plt.xlabel('Features of Houses')
plt.ylabel('Price')
plt.plot(X_test,Y_test,'ro')
plt.show()
