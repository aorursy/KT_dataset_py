import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import pandas as pd

dataset = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
plt.figure()

dataset.hist(figsize=(20, 15), bins=50)

#scatter_matrix(data)

plt.show()
dataset.corr()
new_dataset = dataset.drop(['id', 'date', 'sqft_lot', 'condition', 'yr_built', 'zipcode', 'long', 'sqft_lot15'],axis = 1)

new_dataset
dataset.info()
Y = new_dataset[['price']]

Y
X = new_dataset.drop('price',axis = 1)

X
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print(Y_test)
# from sklearn.preprocessing import StandardScaler

# scalar = StandardScaler()

# #print(np.max(X_train))

# print('before scalling, max is %d and min is %d'%(np.max(np.max(X_train)), np.min(np.min(X_train))))

# X_train2 = scalar.fit_transform(X_train)

# X_test2  = scalar.transform(X_test)  # transform only because that's train

# print('after scalling, max is %d and min is %d'%(np.max(np.max(X_train2)), np.min(np.min(X_train2))))

# # Y_train2 = scalar.fit_transform(Y_train)

# # Y_test2  = scalar.transform(Y_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, Y_train)

print(X_train.shape,Y_train.shape)

model.coef_
prediction = model.predict(X_test)

print(X_test.shape, prediction.shape)

from sklearn.metrics import explained_variance_score

explained_variance_score(Y_test,prediction)
prediction
print(Y_test[:10],' \n ', prediction[:10])
plt.plot(X_test, prediction,'.', X_test, prediction, '.')

plt.title('our first simple model')

plt.xlabel('Features')

plt.ylabel('price')

plt.show()
new_dataset2 =  dataset[['sqft_living']]

new_dataset2
from sklearn.model_selection import train_test_split

new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(new_dataset2, Y, test_size = 0.2, random_state = 0)

# from sklearn.preprocessing import StandardScaler

# scalar = StandardScaler()

# #print(np.max(X_train))

# print('before scalling, max is %d and min is %d'%(np.max(np.max(new_X_train)), np.min(np.min(new_X_train))))

# X_train3 = scalar.fit_transform(new_X_train)

# X_test3  = scalar.transform(new_X_test)  # transform only because that's train

# print('after scalling, max is %d and min is %d'%(np.max(np.max(X_train3)), np.min(np.min(X_train3))))

# Y_train2 = scalar.fit_transform(Y_train)

# # Y_test2  = scalar.transform(Y_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(new_X_train, new_Y_train)

model.coef_
goal_predict = model.predict(new_X_test)

print(new_Y_test[:10],' \n ', goal_predict[:10])

from sklearn.metrics import r2_score

r2_score(new_Y_test,goal_predict )
plt.plot(new_X_test[:10], goal_predict[:10],'.', new_X_test[:10], goal_predict[:10], '-')

plt.title('our first simple model')

plt.xlabel('Features')

plt.ylabel('price')

plt.show()
from sklearn.metrics import explained_variance_score

explained_variance_score(Y_test,goal_predict)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()



# Create The Polynomial Features



from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=3)



x_poly_train = poly_reg.fit_transform(X_train)            

x_poly_test  = poly_reg.fit_transform(X_test) 



# print(x_poly_train.shape,' ', x_poly_test.shape)

# print(Y_train.shape)



regressor.fit(x_poly_train,Y_train)



# Test the model 

y_pred = regressor.predict(x_poly_test)



# Calculate the Accuracy

print('Polynomial Linear Regression Accuracy:',explained_variance_score(Y_test,y_pred))