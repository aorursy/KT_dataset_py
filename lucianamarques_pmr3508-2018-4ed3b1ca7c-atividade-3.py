# data processing, CSV file I/O (e.g. pd.read_csv)
#reference: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
import pandas as pd
import matplotlib.pyplot as np
import matplotlib
import numpy as nm
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#load the train dataset
train_data = pd.read_csv("../input/train.csv")

train_data.head()

median_value = train_data['median_house_value']
median_age = train_data['median_age']

fig, ax= np.subplots()
ax.scatter(median_age, median_value)
total_rooms = train_data['total_rooms']

fig, ax= np.subplots()
ax.scatter(total_rooms, median_value)
total_bedrooms = train_data['total_bedrooms']

fig, ax= np.subplots()
ax.scatter(total_bedrooms, median_value)

population = train_data['population']

fig, ax= np.subplots()
ax.scatter(population, median_value)
households = train_data['households']

fig, ax= np.subplots()
ax.scatter(households, median_value)

median_income = train_data['median_income']

fig, ax= np.subplots()
ax.scatter(median_income, median_value)
#Reference: 
#Matrix of covariance
covar = nm.array(train_data.corr())

#features
features = train_data.columns.values.tolist()

#plotting
figure, axis = np.subplots(figsize=(12,12))
im = axis.imshow(covar)

axis.set_xticks(nm.arange(len(features)))
axis.set_yticks(nm.arange(len(features)))
axis.set_xticklabels(features)
axis.set_yticklabels(features)

np.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor")

#displaying the actual values
for i in range(len(features)):
    for j in range(len(features)):
        text = axis.text(j,i,round(covar[i,j],2),ha="center",va="center",color="w",size=18)
        
#show matrix
np.show()
# Import test data
test_data = pd.read_csv("../input/test.csv")
ID_list = test_data.Id.tolist()

#Create X and Y
train_dataY = train_data['median_house_value']
train_dataX = train_data.drop(columns = ['Id', 'latitude', 'longitude', 'median_house_value'])

regr = linear_model.LinearRegression()

#fit the model
regr.fit(train_dataX, train_dataY)

#print found coefficients
print('Coefficients: \n', regr.coef_)
#Drop Id, latitude and longitude columns
test_data = test_data.drop(columns = ['Id', 'latitude', 'longitude'])

#Predict using coefficients
predict_l = regr.predict(test_data)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()

X = train_dataX
Y = train_dataY

knn.fit(X, Y)

pred_k = knn.predict(test_data)

reg = linear_model.Ridge (alpha = .5)
reg.fit(train_dataX, train_dataY)
reg.coef_
reg.intercept_ 
reg_predicted = reg.predict(test_data)

reg = linear_model.Lasso(alpha = 0.1)
reg.fit(train_dataX, train_dataY)
predicted = reg.predict(test_data)

l = predicted.abs()

my_submission = pd.DataFrame({'Id': ID_list, 'median_house_value': l})
my_submission.to_csv('submission.csv', index=False)

