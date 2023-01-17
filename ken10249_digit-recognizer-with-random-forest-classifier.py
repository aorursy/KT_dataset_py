import numpy as np

import pandas as pd



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

import datetime
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()

# label = prediction
train.columns



# pixel0 -- pixel 783 is 28*28 image
y_train = train['label']

x_train = train.drop(labels = ['label'], axis = 1)
y_train.value_counts()

# balance classification
x_train.isnull().any().any()



# No empty value in dataframe
x_train.max().max()



# range 0 -- 255
def boolean_value(x):

    if (x >= 1):

        x = 1

    else:

        x = 0

    return x

    

for col in x_train.columns:

    x_train[col] = x_train[col].apply(boolean_value)
x_train.max().max()



# 0 for no information

# 1 for has information
parameters = {'n_estimators': [100], 'criterion': ['entropy','gini'], 'max_depth': [20]}



# 50, 10, 5 --> 0.97

# 100, 20, 5 --> 0.995

# 200, 50, 10 --> 



# grid search to perform Hyper-parameter tuning

estimator = RandomForestClassifier(random_state=0)

model = GridSearchCV(estimator = estimator, param_grid = parameters, scoring = 'accuracy', cv = 5)
start_time = datetime.datetime.now()

model.fit(x_train, y_train)

end_time = datetime.datetime.now()
running_time = end_time-start_time
running_time.seconds
predict_train = model.predict(x_train)
from sklearn.metrics import accuracy_score



accuracy_score(y_train,predict_train)

# 0.9549047619047619

# 0.9556190476190476

# 0.9995
test
predict_test = model.predict(test)
submission = pd.read_csv('../input/sample_submission.csv')

submission['Label'] = predict_test
submission