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

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



#Reading the data

X_full = pd.read_csv('../input/titanic/train.csv')

X_test_full = pd.read_csv('../input/titanic/test.csv')





#Target for prediction

y = X_full.Survived



#Chosen features set

data_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']



#Selected from the data

X = X_full[data_features].copy()

X_test = X_test_full[data_features].copy()



    

#Missing values 

#print(X_full.isnull().any())



#sns.scatterplot(x=X_full['Cabin'], y=X_full['Survived'])





#Shape of training data// Number of rows, columns

#print(X.shape)



missing_val_by_columns = (X.isnull().sum())

#print(missing_val_by_columns)





#Training_Test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)





#Get list of categorical variables







object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]











#Making a copy to avoid changing original data

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()

label_X_test = X_test.copy()





# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col]) == set(X_test[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        



print(good_label_cols)



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)

label_X_test = X_test.drop(bad_label_cols, axis=1)







#Apply label encoding

label_encoder = LabelEncoder()



#for col inobject_cols

for col in set(good_label_cols):

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    label_X_test = label_encoder.transform(X_test[col])







#imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(label_X_valid))

#imputer_X_test = pd.DataFrame(my_imputer.transform(label_X_test))





imputed_X_train.columns = label_X_train.columns

imputed_X_valid.columns = label_X_valid.columns

#imputed_X_test.columns = label_X_test.columns



model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)





models = [model_1, model_2, model_3, model_4, model_5]



final_X_test = label_X_test.reshape(1, -1)



# function for comparing different approaches

my_model = RandomForestRegressor(n_estimators=100,random_state=0)

my_model.fit(imputed_X_train, y_train)



my_model.score(imputed_X_train, y_train)

acc_random_forest = round(my_model.score(imputed_X_train, y_train) * 100, 2)

acc_random_forest



Y_pred = my_model.predict(final_X_test)



submission = pd.DataFrame({

    "PassengerId": final_X_test['PassengerId'],

    "Survived": Y_pred

})



submission.to_csv('submissionTita.csv', index=False)


