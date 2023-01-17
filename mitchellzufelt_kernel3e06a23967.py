# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename = "../input/pima-indians-diabetes-database/diabetes.csv"



data = pd.read_csv(filename)
sns.set_style('dark')



sns.distplot(data['Outcome'])
frac = 0.1 #fraction of data that will be reserved for final test data.

test_rows = int(len(data)*frac)

train_data = data[:-test_rows]

test_data = data[-test_rows:]



#We'll hold on to the test data until later. Time to work with the training data.

train_data.describe() #take a look at all those missing values.
#Separate features form target



y = train_data.Outcome



X = train_data.drop('Outcome',axis = 1)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2,random_state=1)
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

imputer = SimpleImputer(missing_values=0.0,strategy='mean')



imp_X_train = pd.DataFrame(imputer.fit_transform(X_train))

imp_X_val = pd.DataFrame(imputer.fit_transform(X_val))



imp_X_train.columns = X_train.columns

imp_X_val.columns = X_val.columns



imp_X_train.head(15) #All missing values have been imputed.
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



for model in [SVC(),RandomForestClassifier(),KNeighborsClassifier(),LogisticRegression()]:

    cvs = cross_val_score(model,X,y,cv=5)

    print(cvs.mean())



#LogisticRegression performed the best, so we will select that as our model. 
model = LogisticRegression()



model.fit(imp_X_train,y_train)

y_preds = model.predict(imp_X_val)



from sklearn.metrics import accuracy_score

accuracy_score(y_val,y_preds)
#We now optimize hyperparameters

from sklearn.model_selection import GridSearchCV



params = {'penalty':['l1','l2','elasticnet','none']}



model_cv = GridSearchCV(model,params,cv=5, scoring = 'accuracy')

model_cv.fit(X,y)



print(model_cv.best_params_)

print(model_cv.best_score_)
model_complete = LogisticRegression(penalty='none')



imp_X = pd.DataFrame(imputer.fit_transform(X))

imp_X.columns = X.columns



y1 = test_data.Outcome



X2 = test_data.drop('Outcome',axis = 1)



model_complete.fit(imp_X,y)

y_pred_fin = model_complete.predict(X2)



accuracy_score(y1,y_pred_fin)