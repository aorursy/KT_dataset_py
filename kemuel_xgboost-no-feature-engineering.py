# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import xgboost

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





train = train.fillna(0)



print(train.iloc[0])
input = ['PassengerId','Pclass', 'Age',	'SibSp','Parch','Fare']

output = 'Survived'

model = xgboost.XGBClassifier()



model.fit(train[input].values,train[output].values)

test = test.fillna(0)

prediction = model.predict(test[input].values)

submission = pd.DataFrame({'PassengerId' : test['PassengerId'].values.astype(int), 'Survived' : prediction.astype(int)})

submission.describe()

submission.to_csv('submissionblindXGB.csv', index=False)