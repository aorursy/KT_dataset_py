#Importing the important libraries for algebra and data processing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # statistical visualization

import matplotlib.pyplot as plt #matlab plots
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)

#Loading data
data = pd.read_csv("../input/student-mat.csv", sep = ';') #Load the clean training data. Splitter is the semi-colon character
data.head()
print ('The data has {0} rows and {1} columns'.format(data.shape[0],data.shape[1]))
data.info()
data.describe() #to look at the numerical fields and their describing mathematical values.
sns.distplot(data['G3']) #Plotting the distribution of the final grades.
corr = data.corr() # only works on numerical variables.
sns.heatmap(corr)
print (corr['G3'].sort_values(ascending=False), '\n')
groupColumns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup'
               , 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

avgColumns = ['G3', 'G2', 'G1']
school = data.groupby(groupColumns[0])[avgColumns].mean()
school.head()
sex = data.groupby(groupColumns[1])[avgColumns].mean()
sex.head()
address = data.groupby(groupColumns[2])[avgColumns].mean()
address.head()
famsize = data.groupby(groupColumns[3])[avgColumns].mean()
famsize.head()
Pstatus = data.groupby(groupColumns[4])[avgColumns].mean()
Pstatus.head()
Mjob = data.groupby(groupColumns[5])[avgColumns].mean()
Mjob.head() #interesting results here. Children of fathers working in the health industry are doing significantly better than children
            #of fathers at home or other.
Fjob = data.groupby(groupColumns[6])[avgColumns].mean()
Fjob.head()
reason = data.groupby(groupColumns[7])[avgColumns].mean()
reason.head()
guardian = data.groupby(groupColumns[8])[avgColumns].mean()
guardian.head()
schoolsup = data.groupby(groupColumns[9])[avgColumns].mean()
schoolsup.head()
famsup = data.groupby(groupColumns[10])[avgColumns].mean()
famsup.head()
paid = data.groupby(groupColumns[11])[avgColumns].mean()
paid.head()
activities = data.groupby(groupColumns[12])[avgColumns].mean()
activities.head()
nursery = data.groupby(groupColumns[13])[avgColumns].mean()
nursery.head()
higher = data.groupby(groupColumns[14])[avgColumns].mean()
higher.head() #another interesting field. 
internet = data.groupby(groupColumns[15])[avgColumns].mean()
internet.head()
romantic = data.groupby(groupColumns[16])[avgColumns].mean()
romantic.head()
focusGroupColumns = ['internet', 'guardian', 'Fjob']
aggs = data.groupby(focusGroupColumns)[avgColumns].mean()
print(aggs.to_string())
X = data.drop('G3', axis=1)
Y = data.G3
X = pd.get_dummies(X) # to convert categorical data to a format that can be used in regression. This isn't the best method to use as it increases the
                      # dimensionality of the dataset but it is a valid place to start
X.info()
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42) # splitting data into 80% test and 20% train since the data is quite small. Usually it's best to use 60:40 or something similar
                                                                                              # with the possibility of validation data for certain types of regression models to avoid overfitting.
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

predicted = regr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_features='auto')
dtr.fit(X_train, Y_train)
predicted = dtr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
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
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))
#Exporting
from sklearn.externals import joblib
#joblib.dump(knn, 'model.pkl') #This will produce a model file that we can import later in a web based python script and possibly take input from a web/mobile application and predict the G3 score