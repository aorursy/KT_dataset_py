# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

pldf = pd.read_csv('../input/openpowerlifting.csv')

feature_columns = ['BodyweightKg','BestBenchKg','BestSquatKg','BestDeadliftKg']
label_column = ['Sex']
#Remove entries with NaN values
pldf.dropna(subset=feature_columns+label_column,inplace=True)

#Calculate basic statistics for feature normalization and for faster convergence
bodyweight = {'mean': pldf['BodyweightKg'].mean(), 'std': pldf['BodyweightKg'].std()}
bestbench = {'mean': pldf['BestBenchKg'].mean(), 'std': pldf['BestBenchKg'].std()}
bestsquat= {'mean': pldf['BestSquatKg'].mean(), 'std': pldf['BestSquatKg'].std()}
bestdeadlift = {'mean': pldf['BestDeadliftKg'].mean(), 'std': pldf['BestDeadliftKg'].std()}

#Feature Normalization
pldf['BodyweightKg'] = (pldf['BodyweightKg'] - bodyweight['mean'])/ bodyweight['std']
pldf['BestBenchKg'] = (pldf['BestBenchKg'] - bestbench['mean'])/ bestbench['std']
pldf['BestSquatKg'] = (pldf['BestSquatKg'] - bestsquat['mean'])/ bestsquat['std']
pldf['BestDeadliftKg'] = (pldf['BestDeadliftKg'] - bestdeadlift['mean'])/ bestdeadlift['std']

#Randomize order of dataset, because it is given ordered
pldf = pldf.reindex(np.random.permutation(pldf.index))

#Split into labels, features for training and testing
msk = np.random.rand(len(pldf)) < 0.8 #80% Training / 20% Testing

pldf_train = pldf[msk]
pldf_test = pldf[~msk]

X_Train = pldf_train[feature_columns]
Y_Train = pldf_train[label_column]
X_Test = pldf_test[feature_columns]
Y_Test = pldf_test[label_column]
#Train Logistic Regression Model for binary classification and unbalanced label (Sex) --> do auto balance
clf = LogisticRegression(solver='newton-cg', max_iter=100, random_state=42,class_weight='balanced').fit(X_Train,Y_Train)
print("training score : %.3f" % (clf.score(X_Train, Y_Train)))

#Evaluate Model by calculating F1 Score
Y_Predictions = pd.DataFrame(data={'Predictions': clf.predict(X_Test)})
f1score = f1_score(y_true=Y_Test,y_pred=Y_Predictions, average='weighted')
print("testing score : %.3f" % (f1score))
# Predict my sex if i did powerlifting using the trained model
me = pd.DataFrame(data={'BodyweightKg': [90.00], 'BestBenchKg': [70.00], 'BestSquatKg': [100.00], 'BestDeadliftKg': [80.00]})
me['BodyweightKg'] = (me['BodyweightKg'] - bodyweight['mean'])/ bodyweight['std']
me['BestBenchKg'] = (me['BestBenchKg'] - bestbench['mean'])/ bestbench['std']
me['BestSquatKg'] = (me['BestSquatKg'] - bestsquat['mean'])/ bestsquat['std']
me['BestDeadliftKg'] = (me['BestDeadliftKg'] - bestdeadlift['mean'])/ bestdeadlift['std']
sex = clf.predict(me)
print(sex)