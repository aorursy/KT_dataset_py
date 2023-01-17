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
import pandas as pd

import numpy as np

from sklearn import svm

import seaborn as sns

from sklearn.metrics import (accuracy_score,mean_squared_error)

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from sklearn import linear_model as lm

from sklearn.model_selection import train_test_split

%matplotlib inline
train_file = "/kaggle/input/eval-lab-2-f464/train.csv"

test_file = "/kaggle/input/eval-lab-2-f464/test.csv"
df_train = pd.read_csv(train_file)

df_test = pd.read_csv(test_file)
df_train.head()
def detail_df(df):

    data_type = pd.concat([df.dtypes,df.nunique(),df.isnull().sum()],axis=1)

    data_type.columns = ["dtype", "unique","no of null"]

    return data_type

df_detail = detail_df(df_train)

df_detail
plt.figure(figsize =(10,10))

sns.heatmap(data = df_train.corr(),annot = True)
drop_columns = []



#dropping columns

df_proc1_train = df_train.drop(labels = drop_columns,axis =1,inplace = False)



df_proc1_test = df_test.drop(labels = drop_columns,axis =1,inplace = False)





numerical_features = ["chem_0","chem_1","chem_4","chem_5","chem_6"]

#categorical_feature = []





X_train = df_proc1_train[numerical_features]

y_train = df_proc1_train["class"]



X_test = df_proc1_test[numerical_features]

#y_train = df_proc1_test["class"]



#scaling

scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_test[numerical_features] = scaler.transform(X_test[numerical_features])
Xt_train,X_val,yt_train,y_val = train_test_split(X_train,y_train,test_size=0.01,random_state=42)
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (accuracy_score,mean_squared_error)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



clf1 = ExtraTreesClassifier()        #Initialize the classifier object



parameters = {

    'max_depth': range(10,25),

    'max_features':['auto'],

    'min_samples_split':[2],

    'n_estimators':[95],



             }    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf1,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(Xt_train,yt_train)        #Fit the gridsearch object with X_train,y_train



best_clf1 = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf1.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf1.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
best_clf1.get_params
np.sqrt(mean_squared_error(clf1.predict(X_val),y_val))
Y_predicted = best_clf1.predict(X_test)

submit = {

    'id':df_proc1_test["id"],

    'class':Y_predicted

}



leng = submit['class'].__len__()



"""

for i in range(leng):

    submit_ran['rating'][i] = int(round(submit_ran['rating'][i]))

"""

df_submit = pd.DataFrame(submit)



save_loc = r"submit.csv"

df_submit.to_csv(save_loc,index=False)
X_test.head()