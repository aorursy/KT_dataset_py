# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('/kaggle/input/learn-together/train.csv',index_col=['Id'])

print(df.shape)

print(df.describe)
cols_with_missing = [col for col in df.columns

                     if df[col].isnull().any()]

print('Columns with missing values:')

print(cols_with_missing)
y=df['Cover_Type']

X=df.drop(['Cover_Type'],axis=1)

print(X.shape,y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42)
sns.distplot(y_train)

y_train.value_counts()
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
numerical_cols = [cname for cname in X_train.columns if 

                X_train[cname].dtype in ['int64', 'float64']]

print(len(numerical_cols))
#from xgboost import XGBClassifier

#from sklearn.metrics import mean_absolute_error

#from sklearn.model_selection import cross_val_score

#from sklearn.metrics import accuracy_score

#from sklearn.model_selection import KFold



#def calc_error(n_estimators):

#    model=XGBClassifier(n_estimators=n_estimators, random_state=0,learning_rate=0.05)

#    model.fit(X_train,y_train)

#    preds = model.predict(X_valid)

#    return(accuracy_score(preds,y_valid))

#pass



#error={}

#for i in range(1,20):

#    error[50*i]=calc_error(50*i)

    

#plt.plot(error.keys(),error.values())
#y_train=y_train-1

#y_train.head()
#y_valid=y_valid-1
import xgboost as xgb

from sklearn.metrics import f1_score,accuracy_score

xgb_clf = xgb.XGBClassifier(objective = "multi:softmax")

# Fit model

xgb_model = xgb_clf.fit(X_train, y_train)

# Predictions

y_train_preds = xgb_model.predict(X_train)

y_valid_preds = xgb_model.predict(X_valid)

# Print F1 scores and Accuracy

print("Training F1 Micro Average: ", f1_score(y_train, y_train_preds, average = "micro"))

print("Valid F1 Micro Average: ", f1_score(y_valid, y_valid_preds, average = "micro"))

print("Valid Accuracy: ", accuracy_score(y_valid, y_valid_preds))
from sklearn.model_selection import RandomizedSearchCV

xgb_clf1 = xgb.XGBClassifier(eval_metric = ['merror','auc'], objective = 'multi:softmax')

# Create parameter grid

params = {"learning_rate": [0.1, 0.3],

               "gamma" : [0.1, 0.3, 0.5],

               "max_depth": [4, 7],

               "colsample_bytree": [0.3, 0.6],

               "subsample": [0.2, 0.5, 0.6],

               "reg_alpha": [0.5, 1],

               "reg_lambda": [1.5, 2, 3.5],

               "min_child_weight": [3, 5],

               "n_estimators": [250, 500]}



# Create RandomizedSearchCV Object

xgb_rscv = RandomizedSearchCV(xgb_clf1, param_distributions = params, scoring = "f1_micro",

                             cv = 5, verbose = 3, random_state = 40)



# Fit the model

xgb_model_tuned = xgb_rscv.fit(X_train, y_train)

best_params=xgb_model_tuned.best_estimator_.get_params()

# Predictions

y_train_preds = xgb_model_tuned.predict(X_train)

y_valid_preds = xgb_model_tuned.predict(X_valid)

# Print F1 scores and Accuracy

print("Training F1 Micro Average: ", f1_score(y_train, y_train_preds, average = "micro"))

print("Valid F1 Micro Average: ", f1_score(y_valid, y_valid_preds, average = "micro"))

print("Valid Accuracy: ", accuracy_score(y_valid, y_valid_preds))
#num_boost_round = model.best_iteration + 1

#print(num_boost_round)

#best_model = xgb.train(

#    params,

#    dtrain,

#    num_boost_round=num_boost_round,

#    evals=[(dtest, "Test")]

#)
X_test=pd.read_csv('/kaggle/input/learn-together/test.csv')

for col in X_test.columns:

    print(col)
X_test=pd.read_csv('/kaggle/input/learn-together/test.csv',index_col=['Id'])

#dpredict = xgb.DMatrix(X_test)

id=X_test.index

preds_test = xgb_model_tuned.predict(X_test)

#preds_test=preds_test+1

out=pd.DataFrame({'Id':id,'Cover_Type':preds_test})

out.head(5)

out.to_csv('submission.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submission.csv')