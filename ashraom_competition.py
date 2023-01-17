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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

my_imputer = Imputer()

file_path="../input/train.csv"
path_test="../input/test.csv"
data = pd.read_csv(file_path)
data_test=pd.read_csv(path_test)

y = data.Survived
features = ['Pclass','Sex', 'Age', 'SibSp','Parch']
X = data[features]

test_X = data_test[features]

one_hot_encoded_X=pd.get_dummies(X)
imputed_X = my_imputer.fit_transform(one_hot_encoded_X)
one_hot_encoded_test_X=pd.get_dummies(test_X)
imputed_test_X = my_imputer.fit_transform(one_hot_encoded_test_X)
#final_train, final_test=one_hot_encoded_X.align(one_hot_encoded_test_X,join='left',axis=1)
#train_X, test_X, train_y, test_y = train_test_split(imputed_X, y, random_state=1)

rf_model =GradientBoostingClassifier()
#rf_model = XGBClassifier(n_estimators=300)
#rf_model = RandomForestClassifier()
#rf_model =BernoulliNB()
# fit your model
rf_model.fit(imputed_X, y)
#rf_predictions = rf_model.predict(imputed_test_X).astype(int)
#rf_model.fit(train_X, train_y, verbose=False)
#rf_model.fit(train_X, train_y)
rf_predictions = rf_model.predict(imputed_test_X)
#print( rf_model.score( val_X,val_y))
#print(accuracy_score(test_y, rf_predictions))


output = pd.DataFrame({'PassengerId':data_test.PassengerId,'survived':rf_predictions})
output.to_csv('submission.csv',index=False)

#plots = plot_partial_dependence(rf_model,features=[0,5],X=imputed_X )



