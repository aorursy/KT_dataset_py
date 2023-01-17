import pandas as pd  #for data handling and wrangling

import numpy as np

import matplotlib.pyplot as plt # for plotting and data visualization

import seaborn as sb # for plotting and data visualization

import sklearn as sk

import random

%matplotlib inline
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn import ensemble

from sklearn.svm import SVC, LinearSVC

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestRegressor
train_data_file_path = './melb_data.csv'

data = pd.read_csv(train_data_file_path)

print(data.head(3))
#Missing Values

data.isnull().sum()
# Percentage of missing values

data.isnull().sum()/len(data)*100
# Remove rows missing data

data = data.dropna()
# present age of the building

data.info()

data['Age'] = 2018 - data['YearBuilt']

data.drop('YearBuilt',axis = 1)

data.to_csv('nona.csv')

#information of dataset after droping not available data
data.columns
#Dropping Columns that are not used in data analysis

#data_new=data.drop(['Bedroom2','Method','Date','SellerG','Postcode','CouncilArea','Propertycount'],axis=1)

cols_to_keep  = [ 'Rooms', 'Distance', 'Landsize',  'Bathroom', 'Car',

       'BuildingArea',   'Lattitude',

       'Longtitude', 'Age','Price']

data_new = data[cols_to_keep]

print(data_new.head(3))
data_new.describe()
# Add age variable based on Year Built

#data_new['Age'] = 2018 - data_new['YearBuilt']

#print(data_new.head())
data_new.to_csv("processed.csv")
data_new['Age'].quantile(q=0.9)
#If we observe the description we can see that there are zeroes in landsize

# and building area which will have an impact on the results.

data_new=data_new[data_new['BuildingArea']!=0]

data_new=data_new[data_new['Landsize']!=0]

data_new=data_new[data_new['Age']<108]

data_new.info()
data_new.columns
print(data_new.head())
decisive_columns = data_new.iloc[:,0:9]

print(decisive_columns.describe())
X = decisive_columns

y = data_new.Price
from sklearn.model_selection import train_test_split

X_train,X_test,train_y,test_y = train_test_split(X,y,test_size=0.2)
# Linear Regression

lr = linear_model.LinearRegression()

lr.fit(X_train, train_y)



# Look at predictions on training and validation set

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



# Plot residuals

plt.scatter(y_train_pred, y_train_pred - train_y, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - test_y, c = "yellow", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 4e6, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, train_y, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, test_y, c = "yellow", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 4e6], [10.5, 4e6], c = "red")

plt.show()
reg = linear_model.LinearRegression()

reg.fit(X_train, train_y)

Y_pred = reg.predict(X_test)

r2 = reg.score(X_test,test_y)

print("R^2 value :" + str(r2) )
print(Y_pred)
reg.score(X_test,test_y)
#Gradient Descent

clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.05, loss = 'ls')

clf.fit(X_train, train_y)



# Look at predictions on training and validation set

y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)



# Plot residuals

plt.scatter(y_train_pred, y_train_pred - train_y, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - test_y, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression after gradient boosting")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 4e6, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, train_y, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, test_y, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression after gradient boosting")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 4e6], [10.5, 4e6], c = "red")

plt.show()
clf.score(X_test,test_y)
# Predicting test set results

y_pred = reg.predict(X_test)

print(y_pred)
# Calculated R Squared

print('R^2 =',metrics.explained_variance_score(test_y,y_pred))
# Actual v predictions scatter

plt.scatter(test_y, y_pred)
#Cofficients

cdf = pd.DataFrame(data= reg.coef_, index = X.columns, columns = ['Coefficients'])

cdf
random.seed(10)

decision_tree = DecisionTreeRegressor()

decision_tree.fit(X_train, train_y)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = decision_tree.score(X_test, test_y) 

print(Y_pred)

acc_decision_tree
#Image(filename='./assets/random_forest.png')
random.seed(10)

random_forests = RandomForestRegressor()

random_forests.fit(X_train, train_y)

Y_pred = random_forests.predict(X_test)

acc_random_forests = random_forests.score(X_test, test_y) 

print(Y_pred)

acc_random_forests
plt.scatter(data_new.BuildingArea,data_new.Price,c=data_new.Distance)
# Get current size

fig_size = plt.rcParams["figure.figsize"]

 

# Prints: [8.0, 6.0]

print ("Current size:", fig_size)

 

# Set figure width to 12 and height to 9

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size
plt.figure(figsize=(10,10))

sb.jointplot(x=data_new.Lattitude, y=data_new.Longtitude, color='green',size=10)

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)
#sb.set_style('darkgrid')

#f, axes = plt.subplots(4,2, figsize = (20,30))



plt.scatter(x = 'Rooms', y = 'Price', data = data_new, edgecolor = 'b',c='Price')

plt.xlabel('Rooms')

plt.ylabel('Price')

plt.title('Rooms v Price')



# Plot [0,1]

plt.scatter(x = 'Distance', y = 'Price', data = data_new, edgecolor = 'b')

plt.xlabel('Distance')

plt.ylabel('Price')

plt.title('Distance v Price')



# Plot [1,0]

plt.scatter(x = 'Bathroom', y = 'Price', data = data_new, edgecolor = 'b')

plt.xlabel('Bathroom')

plt.ylabel('Price')

plt.title('Bathroom v Price')



# Plot [1,1]

plt.scatter(x = 'Car', y = 'Price', data = data_new, edgecolor = 'b')

plt.xlabel('Car')

plt.ylabel('Price')

plt.title('Car v Price')



# Plot [2,0]

plt.scatter(x = 'Landsize', y = 'Price', data = data_new, edgecolor = 'b')

plt.xlabel('Landsize')

plt.ylabel('Price')

plt.title('Landsize v  Price')



# Plot [2,1]

plt.scatter(x = 'BuildingArea', y = 'Price', data = data_new, edgecolor = 'b')

plt.xlabel('BuildingArea')

plt.ylabel('Price')

plt.title('BuildingArea v Price')



# Plot [3,0]

plt.scatter(x = 'Age', y = 'Price', data = data_new, edgecolor = 'b')

plt.xlabel('Age')

plt.ylabel('Price')

plt.title('Age v Price')

plt.figure(figsize=(10,6))

sb.heatmap(data_new.corr(),cmap = 'coolwarm',linewidth = 1,annot= True, annot_kws={"size": 9})

plt.title('Variable Correlation')
from keras.models import Sequential

from keras.layers import Dense
X_train.shape
model = Sequential()



hidden_layer = Dense(1000,input_shape = (9,),activation='tanh')

model.add(hidden_layer)



output_layer = Dense(1,activation='tanh')

model.add(output_layer)



model.compile(metrics= ['accuracy'], optimizer='adam', loss='mean_squared_error')

model.summary()
model.fit(X_train,train_y,epochs=10,batch_size=200)