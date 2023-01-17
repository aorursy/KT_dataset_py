# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

test_path="../input/test.csv"
train_path= "../input/train.csv"

# Any results you write to the current directory are saved as output.
train= pd.read_csv(train_path)
test= pd.read_csv(test_path)
submission= pd.read_csv("../input/gender_submission.csv")
features=['Pclass','Age','SibSp','Parch','Fare']
X= train[features]
y=train.Survived
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=2)
model = XGBClassifier()
model.fit(X_train,y_train)
print(accuracy_score(y_test,model.predict(X_test)))
predictions=model.predict(test[features])
submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission.head()
submission.to_csv('submission.csv',index=False)
