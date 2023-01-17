# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#cross validate

from sklearn import model_selection

from sklearn.model_selection import train_test_split

#ML algorithms

from sklearn.neural_network import MLPClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.describe()
train_df.info()
test_df.describe()
#perform some feature selection, delete columns that only contain zero

all_zero_ = train_df.loc[:,(train_df==0).all(axis=0)] #datafram that contains the columns with all zeros
drop_columns = list(all_zero_) #extract the column names of all_zero_
train_df = train_df.drop(drop_columns, axis=1)

train_df.info() #drop the all_zero_ columns
test_df = test_df.drop(drop_columns, axis=1)

test_df.info() #same with test set
#create train set and label

X = train_df.drop(['label'],axis=1)
#create final test set

X_predict = test_df
#normalize data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_val = X.values

scaler.fit(X_val)

#scalers will transform a dataframe to a nparray???  so create X.values

X_val = scaler.transform(X_val)
column_name = list(X)
#create new dataframe

normX = pd.DataFrame(X_val)

normX.columns = column_name
#do same processing to test data

X_predict_val = X_predict.values

X_predict_val = scaler.transform(X_predict_val)

normX_predict = pd.DataFrame(X_predict_val)

normX_predict.columns = column_name
normX_predict.head()
y = train_df['label']

y.head()
#creating the cross validation sets

#cv_split = model_selection.ShuffleSplit(n_splits = 10, random_state=0)

#cv_results = model_selection.cross_validate(mlp, X_train, y_train, cv=cv_split)
#just do one train test split i guess since it takes too long to run 10 fold

X_train, X_test, y_train, y_test = train_test_split(normX, y, test_size = 0.1)

X_train.describe()
#ML algorithm

mlp = MLPClassifier()
#scoring on train set, validation set

mlp.fit(X_train, y_train)

score_train = mlp.score(X_train, y_train)

score_test = mlp.score(X_test, y_test)

print('score_train is: {}'.format(score_train))

print('score_test is: {}'.format(score_test))
#start hyperparameter tuning

from sklearn.model_selection import GridSearchCV

print("Before hyperparameter tuning:", mlp.get_params())
#parameters that are tuned

#param_grid = {'activation': ['logistic', 'relu'],

#             'alpha': [0.001, 0.0001, 0.00001],

#             'hidden_layer_sizes': [(2,100), (2, 200), (4, 100), (4,200)],

#             'max_iter': [200, 400, 600]}



param_grid = {'alpha': [0.01, 0.001, 0.0001, 0.00001, 0.000001]}
tune_model = GridSearchCV(mlp, param_grid = param_grid, cv=10)

#tune_model = GridSearchCV(mlp, param_grid = param_grid, cv=5)

#how long does this need to run lol
#tune_model.fit(normX, y )

tune_model.fit(X_test, y_test)

print("After hyperparameter tuning: ", tune_model.best_params_)
#accuracy is around 0.92, probably due to using a small sample

print(tune_model.cv_results_['mean_test_score'])
new_mlp = MLPClassifier(alpha=0.01)
new_mlp.fit(X_train, y_train)

score_train = mlp.score(X_train, y_train)

score_test = mlp.score(X_test, y_test)

print('score_train is: {}'.format(score_train))

print('score_test is: {}'.format(score_test))
#result = mlp.predict(normX_predict)

result = new_mlp.predict(normX_predict)

result
#change it to correct submission format

ImageId = list(range(1,28001))

submission = pd.DataFrame({'ImageId': ImageId,

                            'Label': result})
submission.head()
submission.to_csv('submission_csv', index=False)