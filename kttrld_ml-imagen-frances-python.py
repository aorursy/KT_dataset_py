# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import glob

import pathlib

import os

from os.path import isfile, join

mypath = "../input/"

print(os.listdir(mypath))





# Any results you write to the current directory are saved as output.
def getCSV(path, ending):

    csvfiles = [os.path.join(root, name)

             for root, dirs, files in os.walk(path)

             for name in files

             if name.endswith((ending))]

    return(csvfiles)
# exctract the endings to get the projektnames

# this will be used in dictionary

def get_dict(input_csv_files):

    ending = {s.rstrip('0123456789').lower() : files for files in input_csv_files for c in files.columns for s in [c[c.rfind('_')+1:]] if len(s.rstrip('0123456789'))>3 }

    return ending
missing_values = ["nan", np.nan] 

df_data = [pd.read_csv(f, sep=';', decimal=",", na_values = missing_values) for f in getCSV(mypath,".csv")]
# erzeugt ein Dictionary für die Projekte und die Daten

data_dict = get_dict(df_data)

print(data_dict.keys())



projekt = 'IMAGEN' 

#projekt = 'FRANCES'

predict_var = 'C.audit_total_IMAGEN12'

data = data_dict[projekt.lower()]



dataval = data.values

labels = data.columns
df=data

# Get the number of NaN's for each column, discarding those with zero NaN's

ranking = df.loc[:,df.isnull().any()].isnull().sum().sort_values()

# Turn into %

x = ranking.values/len(df)



# Plot bar chart

index = np.arange(len(ranking))

plt.bar(index, x)

plt.xlabel('Features')

plt.ylabel('% NaN observations')

plt.title('% of null data points for each feature of project '+projekt)

plt.show()





print('\nFeatures:',ranking.index.tolist())



data.info()



print('\nData types:',df.dtypes.unique())

print('\nNumber of columns which have any NaN:',df.isnull().any().sum(),'/',len(df.columns))

print('\nNumber of rows which have any NaN:',df.isnull().any(axis=1).sum(),'/',len(df))
n_nans = 5

minfreq = 0.5



# only numeric_data

data_numeric = data.select_dtypes(exclude=['object'])

freq = (data_numeric.isnull().sum())/data_numeric.shape[0]



#reduced_data = data_numeric.loc[:,data_numeric.isnull().sum() < n_nans]

reduced_data = data_numeric.loc[:,freq < minfreq]
df=reduced_data

df.info()



print('\nNumber of columns which have any NaN:',df.isnull().any().sum(),'/',len(df.columns))

print('\nNumber of rows which have any NaN:',df.isnull().any(axis=1).sum(),'/',len(df))
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy = 'mean')

impute_data = reduced_data

data_with_imputed_values_df = pd.DataFrame(my_imputer.fit_transform(impute_data))

data_with_imputed_values_df.columns = impute_data.columns
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



target_data = data_with_imputed_values_df





# Labels are the values we want to predict

labels = np.array(target_data[predict_var])

# Remove the labels from the features

# axis 1 refers to the columns

features= target_data.drop([predict_var, "ID",'C.audit_total_IMAGEN10',"C.audit_freq_IMAGEN10","C.audit_prob_IMAGEN10","C.audit_abuse_flag_IMAGEN10",'C.audit_total_IMAGEN11','C.audit_freq_IMAGEN11',

 'C.audit_abuse_flag_IMAGEN11','C.audit_prob_IMAGEN11','C.audit_symp_IMAGEN11','C.audit_abuse_flag_IMAGEN12',

 'C.audit_freq_IMAGEN12','C.audit_prob_IMAGEN12','C.audit_symp_IMAGEN12'], axis = 1)

# Saving feature names for later use

feature_df = features

feature_list = list(features.columns)

# Convert to numpy array

features = np.array(features)







# z-standardize the features

np.set_printoptions(precision=3)

features = StandardScaler().fit_transform(features)

#labels=(labels - labels.mean(axis=0))/labels.std(dtype = np.float64)

#labels = StandardScaler().fit_transform(labels)

#scaled_X_df = pd.DataFrame(scaled_X, columns=X_labels)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
# Instantiate model with 300 decision trees

rf = RandomForestRegressor(n_estimators = 300, random_state = 0)

# Train the model on training data

rf.fit(train_features, train_labels);
# Get numerical feature importances

importances = list(rf.feature_importances_)

importances2 = rf.feature_importances_

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 7)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

indices = np.argsort(rf.feature_importances_)[::-1]



# Print out the feature and importances 

[print('Variable: {:5} - Importance: {:5}'.format(*pair)) for pair in feature_importances]



for a in range(len(feature_importances)):

    columns = [x[0] for x in feature_importances[:a] if x[1]>= 0.01]

print(columns)

print(len(columns))
import seaborn as sns

sns.set(style="ticks")

sns.pairplot(target_data, vars=columns[:5])
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import svm



# random forest feature selction 

k_list_rf = []

for a in range(5,len(feature_importances)):

    #columns = [x[0] for x in feature_importances[:a]]

    k_list_rf.append([feature_df.columns.get_loc(c) for c in columns if c in feature_df])

k_list_rf 



# Instantiate model with 300 decision trees

rf = RandomForestRegressor(n_estimators = 300, random_state = 0)

svr = svm.SVR(kernel='linear') 

# Train the model on training data

#svr = svm.SVR(kernel='linear') 

features = np.array(feature_df.iloc[:,k_list_rf[0]])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)



rf.fit(train_features, train_labels) 

svr.fit(train_features, train_labels) 



y_pred_rf = rf.predict(test_features)

y_pred_svr = svr.predict(test_features)



MSE_forest = mean_squared_error(test_labels, y_pred_rf)

MSE_svr = mean_squared_error(test_labels, y_pred_svr)

print("The mean squared Error for Random Forest on the training set is: %.2f" % MSE_forest)

print("The mean squared Error on Vector Regression the training set is: %.2f" % MSE_svr)



#Let every single estimator in the tree predict total audit

y_pred_trees = [tree.predict(test_features) for tree in rf.estimators_]





#calculate the standard deviation of the audit

std_audit = np.std(y_pred_trees,axis=0)

# Plot predicted vs real audit on the training set



plt.figure()

plt.title("Predicted vs. Real Audit on Sample \n (standard deviation in 300 trees)")

plt.xlabel("Actual Audit")

plt.ylabel("mean predicted Audit")

plt.errorbar(x=test_labels,y=y_pred_rf,yerr=std_audit, ecolor='y', fmt='b.')

plt.plot(range(0, 20), range(0, 20))

plt.show()
'''from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import svm





# random forest feature selction 

k_list_rf = []

for a in range(5,len(feature_importances)):

    columns = [x[0] for x in feature_importances[:a]]

    k_list_rf.append([feature_df.columns.get_loc(c) for c in columns if c in feature_df])





# random feature selction  

k_list_random = []

while len(k_list_random) < len(k_list_rf[-1]):

    number = np.random.randint(1, len(reduced_data.columns))

    try:

        feature_df.iloc[:,[number]]

        k_list_random.append(number)

        s = set(k_list_random)

        k_list_random=[*s]

    except:

        pass



##########################################################################################################################

##########################################################################################################################



svr = svm.SVR(kernel='linear') 

dic_rf=[]

for a in range(25):

    features = np.array(feature_df.iloc[:,k_list_rf[a]])

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    svr.fit(train_features, train_labels) 

    y_pred_svr = svr.predict(test_features)

    err = mean_squared_error(test_labels, y_pred_svr)

    dic_rf.append(err)



dic_random=[]

for b in range(5,30):

    features2 = np.array(feature_df.iloc[:,k_list_random[:b]])

    train_features, test_features, train_labels, test_labels = train_test_split(features2, labels, test_size = 0.25, random_state = 42)

    svr.fit(train_features, train_labels) 

    y_pred_svr = svr.predict(test_features)

    err = mean_squared_error(test_labels, y_pred_svr)

    dic_random.append(err)

    

##########################################################################################################################

##########################################################################################################################





print ('Random Forest:', dic_rf)    

print ('\nRandom:', dic_random)  

    

index_rf = np.arange(len(dic_rf))

index_random = np.arange(len(dic_random))

plt.plot(index_rf, dic_rf, color='g', label='random Forest')

plt.plot(index_random, dic_random, color='b',label='random Selection')

plt.xlabel('k')

plt.legend(loc='upper right')

plt.ylabel('mean_squared_error')

plt.title('RandomForrest vs. radom feature selection')

plt.show()



nsample = 40

npreds = len(reduced_data.columns)

#ind = sorted(np.random.randint(2, npreds, size=(1, nsample)).tolist()[0])

ind=[ 3,9, 41, 44,  45,  51,  61,  76, 78,  138]'''





'''X_df = reduced_data.loc[:,k_list]

X = X_df.values

X_labels = X_df.columns'''
from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR

#y_pred_svr = svr.predict(X_test)



from sklearn.metrics import mean_absolute_error

#print("The Mean Absolute Error on the training sample: %f" % mean_absolute_error(y_test, y_pred_svr))





X=features

y=labels

scores = []

scores2 = []

best_svr = SVR(kernel='rbf')

cv = KFold(n_splits=10, random_state=42, shuffle=False)

for train_index, test_index in cv.split(X):

    print("Train Index: ", train_index, "\n")

    print("Test Index: ", test_index)



    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    best_svr.fit(X_train, y_train)

    scores.append(best_svr.score(X_test, y_test))

    scores2.append(round(mean_absolute_error(y_test, best_svr.predict(X_test)),2))



print(scores)

print(scores2)







#X = scaled_X_df

#y = data_target_df.values





# create training and testing vars

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

#print( X_train.shape, y_train.shape)

#print( X_test.shape, y_test.shape)
train_features =  X_train

train_labels = y_train

test_features =  X_test

test_labels = y_test



print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import svm



svr = svm.SVR(kernel='linear')

svr.fit(X_train, y_train) 

y_pred_svr = svr.predict(X_test)



from sklearn.metrics import mean_absolute_error

print("The Mean Absolute Error on the training sample: %f" % mean_absolute_error(y_test, y_pred_svr))



regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

y_pred_regr = regr.predict(X_test)



print("The Mean Absolute Error on the training sample: %f" % mean_absolute_error(y_test, y_pred_regr))



# Make predictions using the testing set





'''# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print("Mean squared error: %.2f"

      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_pred))'''





predictions = rf.predict(test_features)

# Calculate the absolute errors

errors = abs(y_pred - y_test)



print(errors)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2))





# Plot outputs

plt.scatter(X_test[:,0], y_test,  color='black')

#plt.plot(X_test, y_pred_svr, color='blue', linewidth=1)

plt.xlabel("Values")

plt.ylabel("Predictions")



plt.show()

# fit a model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)



#print(cross_val_score(lm, X_train, y_train, scoring='r2', cv = 10))
predictions
## The line / model

plt.scatter(y_test, predictions)

plt.xlabel("Values")

plt.ylabel("Predictions")