# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

t = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')


y = df['class']

params = [ 'chem_1','chem_2', 'chem_4', 'chem_6', 'attribute']

x = df[params]

df



#df.corr()



x_test = t

x_test.head()
import xgboost as xgb

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression



from xgboost import XGBClassifier



from sklearn.model_selection import train_test_split
# model = XGBClassifier(n_estimators = 2000)

# model.fit(x,y)
# y_pred = model.predict(x_test[params])

# y_pred = pd.DataFrame(data = y_pred)

# answer = pd.concat([x_test['id'], y_pred], axis = 1)

# answer.columns = ['id', 'class']

# answer.to_csv('lab2_4.csv', index = False)





from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# clf = XGBClassifier()        #Initialize the classifier object



# parameters = {'n_estimators':[100, 200, 300, 400, 500, 600, 700, 1000, 2000],'max_depth':[ 3,5,6,7, 20],'learning_rate':[0.1,0.05,0.2],'n_jobs':[2]} 



# scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(x,y)        #Fit the gridsearch object with X_train,y_train



# best_clf = grid_fit.best_estimator_  

# #Get the best estimator. For this, check documentation of GridSearchCV object



# #unoptimized_predictions = (clf.fit(x, y)).predict(x_test)      #Using the unoptimized classifiers, generate predictions

# optimized_predictions = best_clf.predict(x_test) #Same, but use the best estimator



#print(best_clf)
#optimized_predictions


# optimized_predictions = pd.DataFrame(data = optimized_predictions)

# answer = pd.concat([x_test['id'], optimized_predictions], axis = 1)

# answer.columns = ['id', 'class']

# #answer.to_csv('lab2_3.csv', index = False)

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# model = XGBClassifier(n_estimators = 2000)

# model.fit(X_train,y_train)

# Y_pred = model.predict(X_test)



# from sklearn import metrics

# print(metrics.accuracy_score(y_test,Y_pred))



# etc = ExtraTreesClassifier()

# etc.fit(x,y)
# y_pred = etc.predict(x_test[params])

# y_pred = pd.DataFrame(data = y_pred)

# answer = pd.concat([x_test['id'], y_pred], axis = 1)

# answer.columns = ['id', 'class']

# answer.to_csv('lab2_6.csv', index = False)
model = RandomForestClassifier(n_estimators = 1000)

model.fit(x,y)
y_pred = model.predict(x_test[params])

y_pred = pd.DataFrame(data = y_pred)

answer = pd.concat([x_test['id'], y_pred], axis = 1)

answer.columns = ['id', 'class']

answer.to_csv('lab2_9.csv', index = False)