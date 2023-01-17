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
data=pd.read_csv('../input/train.csv')



data.info()

new_data = data.dropna()

new_data.info()
test_data = pd.read_csv("../input/test.csv")

test_data.info()
new_test_data_dummy = pd.get_dummies(test_data)

columns_list = new_test_data_dummy.columns

print(list(columns_list))


new_data_dummy = pd.get_dummies(new_data)

new_columns = list(columns_list)

new_data_dummy2 = new_data_dummy[new_columns]



new_data_dummy2.shape



new_data_dummy2['income_>50K'] = new_data['income_>50K']
X = new_data_dummy2.drop(['income_>50K'], axis=1)  #  X will hold all features

y = new_data_dummy2['income_>50K']
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 0) 

print(X_train.shape)

print(X_test.shape)
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()



gb_clf.fit(X_train, (y_train))



gb_pred = gb_clf.predict(X_test) 





from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package

acc = accuracy_score(y_test,gb_pred)

print("Accuracy for this model {} %".format(acc*100))
gb_final_pred = gb_clf.predict(new_test_data_dummy) 



x = [ x for x in range(899)]

print(x)





data_to_submit = pd.DataFrame({'id':pd.Series(x),

                               'outcome':pd.Series(gb_final_pred)})



data_to_submit.to_csv('hackathon.csv' , index=False)