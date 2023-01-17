# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing Modules and Cleaning Data

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,f1_score

from sklearn.neighbors import KNeighborsClassifier,kneighbors_graph

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import xgboost as xgb



cam = pd.read_csv('/kaggle/input/credit-card-data/binaryclassifier.csv')

cam.info()

#Finding if any customers were targeted by multiple campaigns

repeat_customers = cam.customer_id.value_counts()

repeat_customers = repeat_customers[repeat_customers > 1]

len(repeat_customers)
# Plotting number of customers contacted by campaign

plot1data = cam.pivot_table(columns = 'branch',index = 'campaign',values = 'customer_id',aggfunc='count')

plot1 = sns.heatmap(plot1data,annot=True,cmap='Greens', fmt='g').set_title("Number of customers contacted by branch per campaign")
plot2data = cam.pivot_table(columns = 'branch',index = 'campaign',values = 'response',aggfunc='sum')

plot2 = sns.heatmap(plot2data,annot=True,cmap='Greens', fmt='g').set_title("Number of customer responses by branch per campaign")
#Cleaning Data and setting Response variables

code_dict = {'M':1,'F':0}

cam.gender = cam.gender.map(code_dict)

X = cam.iloc[:,[0,1,8,9,12,13,14,16,17,18,20,21,22,23,24,25,26,27,28]]

y = cam.response

X = X.set_index(['customer_id','campaign'])



#Splitting Data into training sets and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Hyperparameter Tuning for XGBoost Classification



xgparam_grid = {'max_depth':[3,4,5],'learning_rate':[.1,.2],'n_estimators':[50,100]}

bc = xgb.XGBClassifier(objective="binary:logistic")

searcher = GridSearchCV(estimator = bc,param_grid=xgparam_grid,cv=3,scoring = 'f1')

searcher.fit(X_train,y_train)

searcher.best_params_
searcher.best_score_
#Plotting a confusion matrix for the tuned Classifier



bc = xgb.XGBClassifier(max_depth=5,n_estimators=100,learning_rate=0.1,objective="binary:logistic",scale_pos_weight = 2)

bc.fit(X_train,y_train)

y_pred = bc.predict(X_test)

confusion_matrix(y_test,y_pred)

#Examining ROC-AUC score for goodness of model fit



roc_auc_score(y_test,y_pred)
#Now we take a look at the decision trees the XGBoost model factored in the analysis

xgb.plot_tree(bc)
#Plotting Feature Importances with XGBoost

xgb.plot_importance(bc)
#Setting up a result matrix

Xfinal = X_test.copy()

Xfinal['prediction'] = bc.predict(X_test)

Xfinal = Xfinal.reset_index()





#Plotting Income Distribution of those targeted by Campaigns

plot3=sns.distplot(Xfinal[Xfinal.prediction == 1]['income'],color = 'g',kde_kws = {"label" : 'Responded'})

p1ot3=sns.distplot(Xfinal[Xfinal.prediction == 0]['income'],color = 'r',kde_kws = {"label" : 'Didnt Respond'})

plot3.set_title("Income Distribution of those targeted by Campaigns")
plot4=sns.distplot(Xfinal[Xfinal.prediction == 1]['months_current_account'],color = 'g',kde_kws = {"label" : 'Responded'})

p1ot4=sns.distplot(Xfinal[Xfinal.prediction == 0]['months_current_account'],color = 'r',kde_kws = {"label" : 'Didnt Respond'})

plot4.set_title("Account Distribution of those targeted by Campaigns")
# Number of customers contacted more than once

set1 = set(repeat_customers.index)



#Number of customers who are predicted to respond as per the model

set2 = set(Xfinal[Xfinal.prediction == 1]['customer_id'])



#the customers both sets have in common as a proportion of the 275 respondents the model predicted.



len(set1.intersection(set2))/275
