# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib notebook

# Any results you write to the current directory are saved as output.



from sklearn.model_selection import GridSearchCV
titanic = pd.read_csv('../input/train.csv')
#Wrangling the data set into usable varaibles

lastname = titanic['Name'].str.split(",", n=1, expand=True)

titanic['LastName'] = lastname[0]

titanic['FirstName'] = lastname[1]

titanic['TravelCompanion'] = titanic['Ticket'].duplicated()

titanic['SameLastName'] = titanic['LastName'].duplicated()

titanic['ConfirmedFamily'] = (titanic['TravelCompanion'] == True) & (titanic['SameLastName'] == True)

titanic.drop(columns = ['Name','Ticket', 'Cabin', 'FirstName','LastName', 'PassengerId','SameLastName'], inplace = True)



#Age has ~177 missing values here - do I drop the column, or should I fill it later on?

titanic.dropna(subset=['Embarked','Age'], inplace=True) 

#Age has ~177 missing values here - do I drop the column, or should I fill it later on?



titanic['Pclass'] = titanic['Pclass'].astype(str)

titanic['TravelCompanion'] = titanic['TravelCompanion'].astype(int)

titanic['ConfirmedFamily'] = titanic['ConfirmedFamily'].astype(int)

titanic.head()
#Split Data into x and y

titanic.dtypes

data_X = titanic.drop(columns = 'Survived')

data_Y = titanic['Survived']

data_X = pd.get_dummies(data_X, drop_first=True)

data_X.head()
# Split my Training into training and test data

titanic_train_X, titanic_test_X, titanic_train_Y, titanic_test_Y = train_test_split(data_X, data_Y, 

                                                                                       random_state=37,

                                                                                       train_size = 0.7)
#Find right parameters

# The model you want to set the parameters for

model = DecisionTreeClassifier(class_weight='balanced')



# The parameters to search over for the model

params = {'max_depth':[2,3,4],

          'max_features':['auto','log2',None]}





# Prepare the GridSearch for cross validation

grid_search_dec_tree = GridSearchCV(model, # Note the model is DecisionTreeClassifier as stated above

                                    param_grid=params, # The parameters to search over. 

                                   cv=10, # How many hold out sets to use

                                   n_jobs = 1 # Number of parallel processes to run. 

                                   )



# Do the cross validation on the training data 

grid_search_dec_tree.fit(titanic_train_X, titanic_train_Y)



# Select the best model



best_dec_tree_cv = grid_search_dec_tree.best_estimator_



# Print the best parameter combination 

print(grid_search_dec_tree.best_params_)
# Finally test the performance of the best model on the test data

pred_Y = best_dec_tree_cv.predict(titanic_test_X)



#Print the accuracy 

print(sklmetrics.accuracy_score(titanic_test_Y, pred_Y))

conf_mat = sklmetrics.confusion_matrix(titanic_test_Y, pred_Y)

print(conf_mat)



# Confusion matrix

sns.heatmap(conf_mat, fmt='d',square=True, annot=True, cbar = False, xticklabels = ['Failure','Success'], 

                                                          yticklabels = ['Failure','Success'])

plt.xlabel("Predicted Value")

plt.ylabel("True Value")