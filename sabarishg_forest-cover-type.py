# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv("../input/test.csv")
print(training_data.shape,testing_data.shape)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
training_x = training_data.loc[:, (training_data.columns != 'Cover_Type') & (training_data.columns != 'Id')]
training_y = training_data['Cover_Type']
training_x = pd.Series.as_matrix(training_x)
training_y = pd.Series.as_matrix(training_y)

testing_x = testing_data.loc[:, (testing_data.columns != 'Id')]
testing_x = pd.Series.as_matrix(testing_x)
testid = testing_data['Id']
testid = pd.Series.as_matrix(testid)


print("Training data shape =" + str(training_x.shape))
print("testing data shape =" + str(testing_x.shape))
bagged_classifier = BaggingClassifier(bootstrap=True,max_features=0.7,max_samples=0.8,n_estimators=100,random_state=42)
bagged_scores = cross_val_score(bagged_classifier,X = training_x,y=training_y,cv=5)
print(bagged_scores)
gbm = GradientBoostingClassifier(max_features=0.8,max_depth=4,n_estimators=100,random_state=42,min_samples_split = 10,learning_rate = 0.01)
gbm_scores = cross_val_score(gbm,X = training_x,y=training_y,cv=5)
print(gbm_scores)
log_reg = LogisticRegression(random_state=42,multi_class='multinomial',solver = 'lbfgs',max_iter=10000,n_jobs=-1,penalty='l2')
multinom_scores = cross_val_score(log_reg,X = training_x,y=training_y,cv=5)
multinom_scores
rf = RandomForestClassifier(n_estimators=500,max_depth=15,bootstrap=True)
rf_scores = cross_val_score(rf,X = training_x,y=training_y,cv=5)
rf_scores
#Using the random forest classifier for final predictions
rf.fit(X = training_x,y=training_y)
preds = rf.predict(X=testing_x)

test_preds = pd.DataFrame({"Id": testid,"Cover_Type": preds})
test_preds.to_csv('submission.csv', index=False)
