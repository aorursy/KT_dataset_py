import numpy as np 

import pandas as pd 



df = pd.read_csv("../input/data.csv")

df.head()




del df['Unnamed: 32']

df.head()



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



predictors = list(df.columns.values)

feat = list(predictors)

feat.remove('diagnosis')



X = df[feat]

y = df['diagnosis']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

print("There are {} samples in the training set and {} in the test set".format(X_train.shape[0], X_test.shape[0]))
import xgboost as xgb

xgb_classif = xgb.XGBClassifier()

xgb_classif = xgb_classif.fit(X_train, y_train)



print('The accuracy of the xgboost classifier is {:.2f} out of 1 on the training data'.format(xgb_classif.score(X_train, y_train)))

print('The accuracy of the xgboost classifier is {:.2f} out of 1 on the test data'.format(xgb_classif.score(X_test, y_test)))