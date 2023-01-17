import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.head()
test.head()
test["Survived"] = 0
submission = test.loc[:,["PassengerId", "Survived"]]
submission.head()
submission.to_csv('everyoneDies.csv', index=False)
#Here we can code it as Survived, but if we do so we will overwrite our other prediction. 
#Instead, let's code it as PredGender

test.loc[test['Sex'] == 'male', 'PredGender'] = 0
test.loc[test['Sex'] == 'female', 'PredGender'] = 1
test.PredGender.astype(int)
submission = test.loc[:,['PassengerId', 'PredGender']]
# But we have to change the column name.
# Option 1: submission.columns = ['PassengerId', 'Survived']
# Option 2: Rename command.
submission.rename(columns={'PredGender': 'Survived'}, inplace=True)
submission.to_csv('womenSurvive.csv', index=False)