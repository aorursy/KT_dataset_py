# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#print(train.info())

#print(test.info())



selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

X_train=train[selected_features]

X_test=test[selected_features]



#print(X_train)

#print(X_test)



y_train=train['Survived']



#print(y_train)



X_train['Embarked'].fillna('S',inplace=True)

X_test['Embarked'].fillna('S',inplace=True)



X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)

X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)



from sklearn.feature_extraction import DictVectorizer

dict_vec=DictVectorizer(sparse=False)

X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))

dict_vec.feature_names_



X_test=dict_vec.transform(X_test.to_dict(orient='record'))



from xgboost import XGBClassifier

xgbc=XGBClassifier()



from sklearn.cross_validation import cross_val_score

cross_val_score(xgbc,X_train,y_train,cv=5).mean()



xgbc.fit(X_train,y_train)

xgbc_y_predict=xgbc.predict(X_train)

print(xgbc_y_predict)

from sklearn.metrics import roc_auc_score

#auc=roc_auc_score(label,score)





#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.