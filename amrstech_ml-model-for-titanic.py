

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv('../input/train.csv')

print(data.columns)

print(data.iloc[0])

# Any results you write to the current directory are saved as output.

data['Fare']

data['Fare'].isnull().values.any()
from sklearn.model_selection import train_test_split as tts

from sklearn.impute import SimpleImputer

predictors = ['Pclass','SibSp','Parch','Fare']

X = data[predictors]

y = data['Survived']

train_X, val_X, train_y, val_y = tts(X,y,random_state=0)

from sklearn.ensemble import RandomForestClassifier as RFC

model = RFC(n_estimators = 1000, n_jobs = -1,random_state =40, min_samples_leaf = 2)

model.fit(train_X, train_y)
import numpy as np

test = pd.read_csv('../input/test.csv')

test_X = test[predictors]

l = len(test_X['Fare'])

sum = 0

nan_val = 35.54195598086121

for i in range(l):

    if np.isnan(test_X.iloc[i]['Fare']):

        test_X.loc[i,'Fare'] = nan_val



test_X['Fare'].isnull().values.any()


out = model.predict(test_X)

print(out)

print(model.score(val_X,val_y))
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': out})

my_submission.to_csv('submission.csv',index=False)