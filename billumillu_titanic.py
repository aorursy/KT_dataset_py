# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



X_full = pd.read_csv("../input/train.csv",index_col=0)

X_full_test = pd.read_csv("../input/test.csv",index_col=0)



X_full.shape
X_full.head()
y = X_full.Survived

features = ['Pclass','Sex','Age','SibSp','Parch'] #I had included Cabin as well. Removed it later.

X = X_full[features].copy()

X_test = X_full_test[features].copy()



X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)



#cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

#print(cols_with_missing)

#Only two columns. But how many values are missing?



# --- No need for the above as the below line of code did the task really well!



#X.shape --> (891,6)

X.isnull().sum()

#dropping 'Cabin'

#X_train = X_train.drop("Cabin")

#X_valid = X_train.drop('Cabin',axis=1)



#Error: ['Cabin'] not found in axis ???  --- I'm just gonna remove it from the features itself for now.



#Imputing



X_train.fillna(X_train.mean(),inplace=True) 

#X_train.isnull().any() #dtype changed to bool...

X_valid.fillna(X_valid.mean(),inplace=True)

X_test.fillna(X_valid.mean(),inplace=True)  #Must not forget this!



X_train.head()

from sklearn.preprocessing import LabelEncoder



# Apply label encoder

label_encoder = LabelEncoder()

X_train['Sex'] = label_encoder.fit_transform(X_train['Sex'])

X_valid['Sex'] = label_encoder.transform(X_valid['Sex'])

X_test['Sex'] = label_encoder.fit_transform(X_test['Sex'])



X_train.head() # Female: 0, Male: 1
from xgboost import XGBClassifier



my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)
from sklearn.metrics import accuracy_score



predictions = my_model.predict(X_valid)

print("Accuracy Score: " + str(accuracy_score(predictions, y_valid)))
preds = my_model.predict(X_valid)

preds_test = my_model.predict(X_test)

features = ['Pclass','Sex','Age','SibSp','Embarked']

X = X_full[features].copy()

X_test = X_full_test[features].copy()



# Filling missing values

X.fillna(X.mean(),inplace=True) # So apparently this works only for columns with numbers, not for categorical variables.

X_test.fillna(X_test.mean(),inplace=True)



X['Embarked'] = X['Embarked'].fillna('S')

X_test['Embarked'] = X_test['Embarked'].fillna('S')



#Label Encoding for categorical variables

X['Sex'] = label_encoder.fit_transform(X['Sex'])

X_test['Sex'] = label_encoder.transform(X_test['Sex'])

X['Embarked'] = label_encoder.fit_transform(X['Embarked'])

X_test['Embarked'] = label_encoder.transform(X_test['Embarked'])



#X.isnull().sum()

#X_test.isnull().any()

#X_test.head()



X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)



my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

predictions = my_model.predict(X_valid)

print("Accuracy Score: " + str(accuracy_score(predictions, y_valid)))



preds = my_model.predict(X_valid)

preds_test = my_model.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)