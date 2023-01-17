# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
train1, validate = train_test_split(train, test_size = 0.3, random_state = 100)
train1_y = train1['label']
validate_y = validate['label']
train1_x = train1.drop('label', axis = 1)
validate_x = validate.drop('label', axis = 1)
model_dt = DecisionTreeClassifier(random_state=100)
model_dt.fit(train1_x, train1_y)
validate_pred_dt = model_dt.predict(validate_x)
accuracy_score(validate_y, validate_pred_dt) 
#from sklearn.model_selection import GridSearchCV
#dt_model = RandomForestClassifier()

#params = {'n_estimators': list(range(1,501))}

#tree_cv = GridSearchCV(dt_model, param_grid=params)
#tree_cv.fit(train1_x,train1_y)
#tree_cv.best_params_
model_rf = RandomForestClassifier(random_state=100, n_estimators= 400)
model_rf.fit(train1_x, train1_y)
validate_pred_rf = model_rf.predict(validate_x)
accuracy_score(validate_y, validate_pred_rf)
model_ab = AdaBoostClassifier(random_state=100)
model_ab.fit(train1_x, train1_y)
validate_pred_ab = model_ab.predict(validate_x)
accuracy_score(validate_y, validate_pred_ab)
test_pred_rf = model_rf.predict(test)
prediction = pd.Series(test_pred_rf)
prediction.head()
predict_df = pd.DataFrame(prediction, columns= ['Label'])
predict_df['ImageId'] = test.index + 1
#predict_df

predict_df[['ImageId', 'Label']].to_csv('submission_final.csv', index = False)
pd.read_csv('submission_final.csv')
