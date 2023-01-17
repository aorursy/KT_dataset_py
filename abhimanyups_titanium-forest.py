import os # file directories



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier as RFC # learning: random forests 
# input files: /kaggle/input (read-only) 

# output files: /kaggle/working (max: 5 GB)

# temp files: /kaggle/temp



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# data preview 



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



# add a new column that states if Age is NaN 

train['AgeNA'] = pd.Series(train['Age'].isnull())

test['AgeNA'] = pd.Series(test['Age'].isnull())



# fill in NaN ages with mean age

train = train.fillna(train.mean())

test = test.fillna(test.mean()) 



train.tail()
# gender classification



print(len(train.loc[((train.Sex == "female") & (train.Survived == 1)) | ((train.Sex == "male") & (train.Survived == 0))]) / len(train))
# random forest



pars = ["Pclass", "Sex", "SibSp", "Parch","Age", "AgeNA"] 



y_train = train["Survived"]

X_train = pd.get_dummies(train[pars])

X_test = pd.get_dummies(test[pars])



model = RFC(n_estimators = 100, max_depth = 5)

model.fit(X_train, y_train) 



yhat_train = model.predict(X_train)

yhat_test = model.predict(X_test) 



# train accuracy

print(sum(y_train == yhat_train) / len(y_train)) 



# output 

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': yhat_test})

output.to_csv('submission.csv', index = False) 