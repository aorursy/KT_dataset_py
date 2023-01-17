# for basic operations

import numpy as np 

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# to avoid warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



# for providing path

import os

print(os.listdir("../input"))

# reading the data

data = pd.read_csv('../input/uci-secom.csv')



# getting the shape of the data

print(data.shape)

# getting the head of the data



data.head()
# checking if the dataset contains any NULL values



data.isnull().any().any()
# Replacing all the NaN values with 0 as the values correspond to the test results.

# since, the values are not present that means the values are not available or calculated

# so better we not take median or mean and replace them with zeros



data = data.replace(np.NaN, 0)



# again, checking if there is any NULL values left

data.isnull().any().any()
# distribution plot to see first four column's distribution



plt.rcParams['figure.figsize'] = (18, 16)



plt.subplot(2, 2, 1)

sns.distplot(data['1'], color = 'darkblue')

plt.title('First Sensor Measurements', fontsize = 20)



plt.subplot(2, 2, 2)

sns.distplot(data['2'], color = 'red')

plt.title('Second Sensor Measurements', fontsize = 20)



plt.subplot(2, 2, 3)

sns.distplot(data['3'], color = 'darkgreen')

plt.title('Third Sensor Measurements', fontsize = 20)



plt.subplot(2, 2, 4)

sns.distplot(data['4'], color = 'yellow')

plt.title('Fourth Sensor Measurements', fontsize = 20)



plt.show()
# pie chart



labels = ['Fail', 'Pass']

size = data['Pass/Fail'].value_counts()

colors = ['magenta', 'green']

explode = [0, 0.1]



plt.style.use('seaborn-deep')

plt.rcParams['figure.figsize'] = (8, 8)

plt.pie(size, labels =labels, colors = colors, explode = explode, autopct = "%.2f%%", shadow = True)

plt.axis('off')

plt.title('Target: Pass or Fail', fontsize = 20)

plt.legend()

plt.show()
# heatmap



plt.rcParams['figure.figsize'] = (18, 18)

sns.heatmap(data.corr(), cmap = 'copper')

plt.title('Heatmap for the Data', fontsize = 20)
# deleting the first column



data = data.drop(columns = ['Time'], axis = 1)



# checking the shape of the data after deleting a column

data.shape
# separating the dependent and independent data



x = data.iloc[:,:590]

y = data.iloc[:, 590]



# getting the shapes of new data sets x and y

print("shape of x:", x.shape)

print("shape of y:", y.shape)
# splitting them into train test and split



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



# gettiing the shapes

print("shape of x_train: ", x_train.shape)

print("shape of x_test: ", x_test.shape)

print("shape of y_train: ", y_train.shape)

print("shape of y_test: ", y_test.shape)
# standardization



from sklearn.preprocessing import StandardScaler



# creating a standard scaler

sc = StandardScaler()



# fitting independent data to the model

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

# Under Sampling



failed_tests = np.array(data[data['Pass/Fail'] == 1].index)

no_failed_tests = len(failed_tests)



print(no_failed_tests)
normal_indices = data[data['Pass/Fail'] == -1]

no_normal_indices = len(normal_indices)



print(no_normal_indices)
random_normal_indices = np.random.choice(no_normal_indices, size = no_failed_tests, replace = True)

random_normal_indices = np.array(random_normal_indices)



print(len(random_normal_indices))
under_sample = np.concatenate([failed_tests, random_normal_indices])

print(len(under_sample))
# creating the undersample data



undersample_data = data.iloc[under_sample, :]


# splitting the undersample dataset into x and y sets



x = undersample_data.iloc[:, undersample_data.columns != 'Pass/Fail'] 

y = undersample_data.iloc[:, undersample_data.columns == 'Pass/Fail']



print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



x_train_us, x_test_us, y_train_us, y_test_us = train_test_split(x, y, test_size = 0.2, random_state = 0)



print(x_train_us.shape)

print(y_train_us.shape)

print(x_test_us.shape)

print(y_test_us.shape)
# standardization



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train_us)

x_test = sc.transform(x_test_us)
import xgboost as xgb

from xgboost.sklearn import XGBClassifier



model = XGBClassifier()



model.fit(x_train_us, y_train_us)



y_pred = model.predict(x_test_us)
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test_us, y_pred)





plt.rcParams['figure.figsize'] = (5, 5)

sns.set(style = 'dark', font_scale = 1.4)

sns.heatmap(cm, annot = True, annot_kws = {"size": 15})



# It is able to predict 17 defected semiconductors among 21 Semi-Conductors
# Applying Grid Search CV to find the best model with the best parameters



from sklearn.model_selection import GridSearchCV



parameters = [{'max_depth' : [1, 2, 3, 4, 5, 6]}]



grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = 2, n_jobs = -1)



grid_search = grid_search.fit(x_train_us, y_train_us)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
print("Best Accuracy: ", best_accuracy)

print("Best Parameter: ", best_parameters)
import xgboost as xgb

from xgboost.sklearn import XGBClassifier



weights = (y == 0).sum()/(1.0*(y == -1).sum())

model = XGBClassifier(max_depth = 4, scale_pos_weights = weights, n_jobs = 4)



model.fit(x_train_us, y_train_us)



y_pred = model.predict(x_test_us)
# plotting the feature importances



colors = plt.cm.spring(np.linspace(0, 1, 9))

xgb.plot_importance(model, height = 1, color = colors, grid = True, importance_type = 'cover', show_values = False)



plt.rcParams['figure.figsize'] = (18, 20)

plt.xlabel('The F-Score for each features')

plt.ylabel('Importances')

plt.show()
from imblearn.over_sampling import SMOTE



x_resample, y_resample  = SMOTE().fit_sample(x, y.values.ravel())



print(x_resample.shape)

print(y_resample.shape)
from sklearn.model_selection import train_test_split



x_train_os, x_test_os, y_train_os, y_test_os = train_test_split(x, y, test_size = 0.2, random_state = 0)



print(x_train_os.shape)

print(y_train_os.shape)

print(x_test_os.shape)

print(y_test_os.shape)
# standardization



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train_os = sc.fit_transform(x_train_os)

x_test_os = sc.transform(x_test_os)
import xgboost as xgb

from xgboost.sklearn import XGBClassifier



model = XGBClassifier()



model.fit(x_train_os, y_train_os)



y_pred = model.predict(x_test_os)
# Applying Grid Search CV to find the best model with the best parameters



from sklearn.model_selection import GridSearchCV



# making a parameters list

parameters = [{'max_depth' : [1, 10, 5, 7]}]



# making a grid search model

grid_search = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = 2, n_jobs = -1)

grid_search = grid_search.fit(x_train_os, y_train_os)



# getting the results

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print("Best Accuracy: ", best_accuracy)

print("Best Parameter: ", best_parameters)
import xgboost as xgb

from xgboost.sklearn import XGBClassifier



weights = (y == 0).sum()/(1.0*(y == -1).sum())

model = XGBClassifier(max_depth = 10, scale_pos_weights = weights, n_jobs = 4)



model.fit(x_train_os, y_train_os)



y_pred = model.predict(x_test_os)

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test_os, y_pred)





plt.rcParams['figure.figsize'] = (5, 5)

sns.set(style = 'dark', font_scale = 1.4)

sns.heatmap(cm, annot = True, annot_kws = {"size": 15}, cmap = 'spring')
# again creating x and y from the dataset



x = data.iloc[:, :-1]

y = data.iloc[:, -1]



# getting the shapes

print("Shape of x:", x.shape)

print("Shape of y:", y.shape)
# splittng the dataset



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



# print the shapes

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
# defining outlier fraction



Fraud = data[data['Pass/Fail']==1]

Valid = data[data['Pass/Fail']==-1]



outlier_fraction = len(Fraud)/float(len(Valid))

print("Outlier Fraction :", outlier_fraction)
from sklearn.ensemble import IsolationForest



model = IsolationForest(n_estimators=100, max_samples=len(x_train), 

                                       contamination=outlier_fraction, random_state=0, verbose=0)

model.fit(x_train, y_train)

scores_prediction = model.decision_function(x_train)

y_pred = model.predict(x_test)
# evaluating the model



# printing the confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, cmap = 'rainbow')
from sklearn.neighbors import LocalOutlierFactor





model = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, 

                           metric_params=None, contamination=outlier_fraction)



model.fit(x_train, y_train)

y_pred = model.fit_predict(x_test)

# evaluating the model

# printing the confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, cmap = 'summer')

from sklearn.svm import OneClassSVM



model = OneClassSVM(kernel ='rbf', degree=3, gamma=0.1,nu=0.005, max_iter=-1, random_state=0)



model.fit(x_train, y_train)

y_pred = model.fit_predict(x_test)



# evaluating the model

# printing the confusion matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm ,annot = True, cmap = 'winter')

Recall = np.array([77.3, 77.3, 84.6, 84.6, 61.5])

label = np.array(['UnderSampling', 'OverSampling', 'IsolationForest', 'LocalOutlier', 'OneClassSVM'])

indices = np.argsort(Recall)

color = plt.cm.rainbow(np.linspace(0, 1, 9))



plt.rcParams['figure.figsize'] = (18, 7)

plt.bar(range(len(indices)), Recall[indices], color = color)

plt.xticks(range(len(indices)), label[indices])

plt.title('Recall Accuracy', fontsize = 30)

plt.grid()

plt.tight_layout()

plt.show()