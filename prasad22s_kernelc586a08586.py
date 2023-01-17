# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
test = pd.read_csv("../input/test.csv")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
train.shape
train, val = train_test_split(train, test_size = 0.3, random_state = 100)

train_x = train.drop('label', axis = 1)
train_y = train['label']
val_x = val.drop('label', axis = 1)
val_y = val['label']
#model_dt = DecisionTreeClassifier(random_state=100, max_depth=5)
#model_dt.fit(train_x, train_y)
#pred_test = model_dt.predict(val_x)
#pred_results = pd.DataFrame({
#    'actual':val_y, 
#    'predicted':pred_test
#})
#accuracy_score(pred_results['actual'], pred_results['predicted'])


#params = {'max_depth': list(range(1, 25))}
#base_estimator = DecisionTreeClassifier(random_state=100)

#model = GridSearchCV(base_estimator, param_grid = params)
#model.fit(train_x,train_y)
#model.best_params_ = 15
model_dt = DecisionTreeClassifier(random_state=100, max_depth=15)
model_dt.fit(train_x, train_y)
pred_test = model_dt.predict(val_x)
pred_results = pd.DataFrame({
    'actual':val_y, 
    'predicted':pred_test
})
ac_dt = accuracy_score(pred_results['actual'], pred_results['predicted'])
print(ac_dt) #accuracy score of decision tree
model_rf = RandomForestClassifier(random_state=100,
                                 n_estimators=300)
model_rf.fit(train_x,train_y)
pred = model_rf.predict(val_x)
ac_rf = accuracy_score(val_y, pred)
print(ac_rf) #accuracy of Random Forest
model_ab = AdaBoostClassifier(random_state=100, n_estimators=300)
model_ab.fit(train_x, train_y)
pred_test = model_ab.predict(val_x)
ac_ab = accuracy_score(val_y, pred_test)
print(ac_ab) #accuracy of AdaBoost
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(train_x, train_y)
pred_class = model_knn.predict(val_x)
ac_knn = accuracy_score(val_y, pred_class)
print(ac_knn) #Acuracy of KNN
y_final=model_knn.predict(test)
solution = pd.DataFrame({"ImageId":test.index+1,"Label":y_final})
solution.to_csv("Digit_Recognizer_KNN.csv",index=False)

