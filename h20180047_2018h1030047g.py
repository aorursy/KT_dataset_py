import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor







from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score



%matplotlib inline
df = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
df.fillna(value=df.mean(),inplace=True)
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df['type'] = le.fit_transform(df['type'])

df.head()
corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10,10))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
categorical_features = ['type']



numerical_features = df.drop(['rating', 'type', 'id',], axis = 1)

#numerical_features = ['feature6', 'feature8','feature4','feature2','feature11']

numerical_features = list(numerical_features.columns)





X = df[numerical_features]# + categorical_features]

#X = df[numerical_features]

y = df['rating']

X.head()
categorical_features = ['type']



numerical_features = df.drop(['rating', 'type', 'id',], axis = 1)

#numerical_features = ['feature6', 'feature8','feature4','feature2','feature11']

numerical_features = list(numerical_features.columns)





X = df[numerical_features]# + categorical_features]

#X = df[numerical_features]

y = df['rating']

X.head()
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)  #Checkout what does random_state do

# clf = RandomForestClassifier(n_estimators = 300)        #Initialize the classifier object



# parameters = {'n_estimators':[250,300,350],

#               #'max_depth':[5,10,15,20,28,30,40,50,60,70,80,90]

#              }    #Dictionary of parameters



# scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(X,y)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



# # unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_test)      #Using the unoptimized classifiers, generate predictions

# # optimized_predictions = best_clf.predict(X_test)        #Same, but use the best estimator



# # acc_unop = accuracy_score(y_test, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

# # acc_op = accuracy_score(y_test, optimized_predictions)*100         #Calculate accuracy for optimized model

# # print("Accuracy score on unoptimized model:{}".format(acc_unop))

# # print("Accuracy score on optimized model:{}".format(acc_op))
#2nd grid search



clf = RandomForestClassifier()        #Initialize the classifier object

clf1 = RandomForestClassifier(n_estimators = 300)

parameters = {'n_estimators':[250,300,350],

              #'max_depth':[10, 15, 20, 30, 50, 60, 70, 80, 90],

              #'max_features': ['auto', 'sqrt', 'log2'],

              #'criterion' :['gini', 'entropy']

             }  



scorer = make_scorer(accuracy_score)         



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         



grid_fit = grid_obj.fit(X_train,y_train)    



best_clf = grid_fit.best_estimator_         

# clf1.fit(X_train,y_train)

unoptimized_predictions = (clf1.fit(X_train, y_train)).predict(X_test)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_test)        #Same, but use the best estimator



acc_unop = accuracy_score(y_test, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_test, optimized_predictions)*100         #Calculate accuracy for optimized model

print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
from sklearn.metrics import mean_squared_error

from math import sqrt



mae_lr = sqrt(mean_squared_error(optimized_predictions,y_test))



print("Mean Absolute Error of Linear Regression: {}".format(mae_lr))
test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
test.fillna(value=test.mean(),inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



test['type'] = le.fit_transform(test['type'])
X_test = test[numerical_features]# + categorical_features]

y_pred = best_clf.predict(X_test)
var = pd.DataFrame({'id': test['id'], 'rating': y_pred})

var.head()
var.to_csv("Submit8.csv", index = False)