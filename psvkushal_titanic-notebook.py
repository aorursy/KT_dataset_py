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
#importing the data

comp_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

comp_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
comp_train_data.head(1)
comp_test_data.head(1)
#finding the number of nan in each columns

comp_train_data.isna().sum()
#preproceesing the train data

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

drop_cols = ['Age','Cabin','Name','Ticket']

drop_train_data = comp_train_data.drop(drop_cols,axis=1)

drop_train_data = drop_train_data[True ^ drop_train_data['Embarked'].isna()]



procc_y_train = drop_train_data['Survived']

drop_X_train = drop_train_data.drop('Survived',axis=1)



cate_col = ['Sex','Embarked']

#implementing onehotencoder on the data

Embark_coder = OneHotEncoder(sparse=False)

#since the output will be in the form of numpy array should convert it into the form of dataframe

cate_X_train =pd.DataFrame(Embark_coder.fit_transform(drop_X_train[cate_col]))

cate_X_train.index = drop_X_train.index

num_X_train = drop_X_train.drop(cate_col,axis=1)

procc_X_train = pd.concat([cate_X_train,num_X_train],axis=1)



#preproccessing the test data

missing_values = SimpleImputer()

drop_X_test = comp_test_data.drop(drop_cols,axis=1)

cate_X_test = pd.DataFrame(Embark_coder.transform(drop_X_test[cate_col]))

cate_X_test.index = drop_X_test.index

miss_col = list(drop_X_test.columns[drop_X_test.isna().sum() != 0])

num_X_test = drop_X_test.drop(cate_col+miss_col,axis=1)

miss_X_test = pd.DataFrame(missing_values.fit_transform(drop_X_test[miss_col]),columns=miss_col)

procc_X_test = pd.concat([cate_X_test,num_X_test,miss_X_test],axis=1)

print("the categories used for one hot encoding are : ")

print(Embark_coder.categories_)
miss_X_test = pd.DataFrame(missing_values.fit_transform(drop_X_test[miss_col]),columns=miss_col)

miss_X_test
#splitting the data for training and validation

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(procc_X_train, procc_y_train)
set(X_train.columns)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

#ignore cols

ignore_cols = ['PassengerId']

req_cols = set(X_train.columns)-set(ignore_cols)

#creation of the model

logistic_model = LogisticRegression(random_state=3,max_iter=5e3)



#training of the model

logistic_model.fit(X_train[req_cols],y_train)



#prediciting the model

y_pred = logistic_model.predict(X_val[req_cols])

acc = accuracy_score(y_pred,y_val)

print("accuracy of the logistic regression model is {}".format(acc))
drop_X_train
print("The parameters for logistic regression model are")

print(logistic_model.coef_)
#predicitons for test data

y_pred_test = logistic_model.predict(procc_X_test[req_cols])

output = pd.DataFrame({"PassengerId":procc_X_test["PassengerId"],'Survived':y_pred_test})

output.to_csv('my_submission.csv',index=False)
len(y_pred_test)