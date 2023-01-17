# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))





"""

One-hot encoding for the sex and everything else should be fine since all other values are numeric

we will use [pclass,Sex,Age,SibSp,ParCh] as out features

we are predicting survival so [Survived will be our label]

"""

file_train = pd.read_csv("../input/train.csv")

file_test = pd.read_csv("../input/test.csv")



analyse_gender_file = pd.read_csv("../input/gender_submission.csv")



data_train = pd.DataFrame(file_train)

data_test = pd.DataFrame(file_test)



#data_gender = pd.DataFrame(analyse_gender_file)



feature_columns = ['Pclass','Sex','Age','SibSp','Parch']



train_y = data_train.Survived

train_X = data_train[feature_columns]

test_X = data_test[feature_columns]



s = (train_X.dtypes=='object')

obj_cols = list(s[s].index)

#Applying one-hot encoder



OH_sex = OneHotEncoder(handle_unknown = 'ignore', sparse=False)



OH_feature_columns = pd.DataFrame(OH_sex.fit_transform(train_X[obj_cols]))

OH_test_feature_columns = pd.DataFrame(OH_sex.transform(test_X[obj_cols]))



OH_feature_columns.index = train_X.index

OH_test_feature_columns.index = test_X.index



num_X_train = train_X.drop(obj_cols,axis=1)

num_X_test = test_X.drop(obj_cols,axis=1)



OH_X_train = pd.concat([num_X_train,OH_feature_columns], axis=1)

OH_text_X = pd.concat([num_X_test,OH_test_feature_columns],axis = 1)





#OH_X_train.sample(10)



#imputing so we replace the Nan values and all

my_imputer = SimpleImputer()



Imputed_X_train = pd.DataFrame(my_imputer.fit_transform(OH_X_train))

Imputed_X_test = pd.DataFrame(my_imputer.fit_transform(OH_text_X))



Imputed_X_train.columns = OH_X_train.columns



Imputed_X_test.columns = OH_text_X.columns







final_X_train, final_X_Valid, final_y_train, final_y_valid = train_test_split(Imputed_X_train,train_y,test_size=0.2,random_state=1)



model = SGDClassifier()

model.fit(final_X_train,final_y_train)



evaluater= model.predict(Imputed_X_test)



indeces = []

model_predictions = []

for n in range(evaluater.shape[0]):

    indeces.append(n+892)

    

for n in evaluater:

    model_predictions.append(np.argmax(n))

    

#print(model_predictions)



output = pd.DataFrame({'PassengerId': indeces,

                       'Survived':evaluater})

output.to_csv('submission.csv', index=False)

#analyse_gender_file

output



# Any results you write to the current directory are saved as output.