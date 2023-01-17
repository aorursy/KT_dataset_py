#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
train = pd.read_csv("../input/car-prices/data.csv") #Load the clean training data
print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
#check missing values
train.columns[train.isnull().any()]
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))
#create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
nd = pd.melt(train, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')

#correlation plot
corr = numeric_data.corr()
sns.heatmap(corr)
print (corr['Price'].sort_values(ascending=False), '\n')
cat_data.describe()
sp_pivot = train.pivot_table(index='Make', values='Price', aggfunc=np.mean).sort_values(by='Price') #Get mean price per make
sp_pivot.plot(kind='bar',color='blue')
#GrLivArea variable
sns.jointplot(x=np.log(train['Mileage']), y=np.log(train['Price']))
X = train[['Year', 'Mileage', 'Make', 'State']] #Model emitted as there are thousands of models and XGB will not converge with that many features
Y = train.Price
X = pd.get_dummies(data=X)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)
#gradient booster with one hot conversion
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='ls', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
data = pd.read_csv("../input/carpredictiondata/dataProc.csv") #Load the processed data. Categorical features are converted to numerical ones
                                                              #, and outliers are removed
#plotting corr of the data we plan to use
visData = data[['Year', 'Mileage', 'MakeNum', 'StateNum', 'ModelNum', 'Price']] 
corr = visData.corr()
sns.heatmap(corr)
print (corr['Price'].sort_values(ascending=False), '\n')
X = data[['Year', 'Mileage', 'MakeNum', 'StateNum', 'ModelNum']]
Y = data.Price

#Split into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)
X_test.info
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))

# Uncomment the next 2 lines to produce a model file that you can use later
#from sklearn.externals import joblib
#joblib.dump(knn, 'model.pkl')
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_features='auto')
dtr.fit(X_train, Y_train)
predicted = dtr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)


plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

predicted = regr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
sns.distplot(data.Price)
sns.distplot(data.Mileage)
X = data[['Year', 'Mileage', 'MakeNum', 'StateNum', 'ModelNum']] #CityNum is state, misnamed in data
Y = np.log(data.Price)
X['Mileage'] = np.log(X['Mileage'])
X['Mileage'] = 0.9 * X['Mileage'] + 0.1

#Split into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)
sns.distplot(Y)
sns.distplot(X['Mileage'])
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
err = Y_test - predicted

p2 = knn.predict(X_train)
err2 = Y_train - p2

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
#Compare error of training data vs error of test data. We need to get the exp of the data since we used log before
xperr = np.exp(Y_test) - np.exp(predicted)
xperr2 = np.exp(Y_train) - np.exp(p2)
sns.distplot(xperr)
sns.distplot(xperr2)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_train, p2))
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))