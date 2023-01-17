import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as sns 

from sklearn.model_selection import train_test_split 

from sklearn import metrics

from imblearn.over_sampling import SMOTE



%matplotlib inline
train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

numerical_features = ['chem_0', 'chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

X = train[numerical_features]

X.head()

train.describe()
y = train['class']

y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# from sklearn.preprocessing import RobustScaler

# from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import MinMaxScaler





# # scaler = RobustScaler()

# # scaler = StandardScaler()

# scaler = MinMaxScaler()





# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

# X_test[numerical_features] = scaler.transform(X_test[numerical_features])  



# # It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc



# X_train[numerical_features].head()
# sm = SMOTE(ratio = 'minority', k_neighbors = 3)

# x_train_res, y_train_res = sm.fit_sample(X, y)

from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)



print(X_ros.shape[0] - X.shape[0], 'new random picked points')
y_ros.size
df1 = pd.DataFrame(X_ros, columns = numerical_features)

df1.head()
df2 = pd.DataFrame(y_ros, columns = ['class'])

df2
new_train = pd.concat([df1, df2], axis = 1)
new_train.head()
new_train['class'].value_counts()
numerical_features = ['chem_0', 'chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

X = new_train[numerical_features]

X.head()
y = new_train['class']

y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_train = X

# y_train = y
from sklearn.ensemble import ExtraTreesClassifier

clf2 = ExtraTreesClassifier(n_estimators = 200).fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred_2 = clf2.predict(X_test)

acc2 = accuracy_score(y_pred_2,y_test)*100

print("Accuracy score of clf2: {}".format(acc2))
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import make_scorer



# #TODO

# clf = ExtraTreesClassifier(n_estimators = 200).fit(X_train,y_train)        #Initialize the classifier object



# parameters = {'n_estimators':[50,100,150,200,250,300,350,400,450,500],

#              'max_depth': [10,20,30,35,40,45,50,60,70, None]}    #Dictionary of parameters



# scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



# unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_test)      #Using the unoptimized classifiers, generate predictions

# optimized_predictions = best_clf.predict(X_test)        #Same, but use the best estimator



# acc_unop = accuracy_score(y_test, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

# acc_op = accuracy_score(y_test, optimized_predictions)*100         #Calculate accuracy for optimized model



# print("Accuracy score on unoptimized model:{}".format(acc_unop))

# print("Accuracy score on optimized model:{}".format(acc_op))
# grid_fit.best_estimator_  
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(objective='multiclass', learning_rate = 0.3)

lgbm.fit(X_train, y_train)

y_predx = lgbm.predict(X_test)

acc2 = accuracy_score(y_predx,y_test)*100

print("Accuracy score of clf2: {}".format(acc2))
y_predx
test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

X_test = test[numerical_features]

X_test
pred = lgbm.predict(X_test)
pred
var = pd.DataFrame({'id': test['id'], 'class': pred})

var
var.to_csv("Submit_lgbm2.csv", index = False)