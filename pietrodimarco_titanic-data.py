# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from xgboost import XGBClassifier


# Any results you write to the current directory are saved as output.
test=pd.read_csv("../input/test.csv")
train=pd.read_csv("../input/train.csv")
tot=pd.concat([test,train])
for i,row in tot.iterrows():
    if(np.isnan(row['Age'])):
        tot.set_value(i,'Age',tot[tot['Pclass']==row['Pclass']]['Age'].mean()) 
tot=tot.drop(['Name','Ticket'],axis=1)
tot=pd.get_dummies(tot)
df=pd.DataFrame(columns=tot.columns.values)
dfTest=pd.DataFrame(columns=tot.columns.values)
for i,row in tot.iterrows():
    if(row['PassengerId'] in train['PassengerId']):
        df=df.append(row)
    else:
        dfTest=dfTest.append(row)
df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

y=df['Survived']
X=df.drop(['Survived'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
my_imp=Imputer()
train_X=my_imp.fit_transform(train_X)
test_X=my_imp.transform(test_X)
my_model = XGBClassifier(n_estimators=500)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
from sklearn.metrics import accuracy_score
# make predictions
predictions = my_model.predict(test_X)
accuracy_score(test_y, predictions)

dfTest=dfTest.drop(dfTest[dfTest.Survived.notnull()].index)

data_to_submit = pd.DataFrame({
    'PassengerId':dfTest['PassengerId'].astype(int),
    'Survived': my_model.predict(dfTest.drop(['Survived'],axis=1).as_matrix()).astype(int)
})
data_to_submit
data_to_submit.to_csv('csv_to_submit.csv', index = False)