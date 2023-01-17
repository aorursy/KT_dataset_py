# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


x = pd.read_csv('../input/train.csv')


xtest = pd.read_csv('../input/test.csv')
x.head()
filterd_x.isna().sum()
filterd_x = x.drop(columns = ["Pclass", "SibSp","Parch","Ticket","Fare","Cabin","Name", "Embarked"], axis = 1)
##age_filled = filterd_x['Age'].fillna((filterd_x['Age'].mean()), inplace=True)
mean_age = filterd_x['Age'].mean()
filterd_x['Age'] = filterd_x['Age'].fillna(mean_age)


test_x = xtest.drop(columns = ["Pclass", "SibSp","Parch","Ticket","Fare","Cabin","Name", "Embarked"], axis = 1)
mean_age_test = xtest['Age'].mean()
test_x['Age'] = xtest['Age'].fillna(mean_age_test)
#filterd_x.count()

################

genders = {"male": 0, "female": 1}
data = [filterd_x, test_x]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
age_filled = filterd_x.Age.values
sex = filterd_x.Sex.values
y = filterd_x.Survived.values
x.head(5)
age_filled.shape
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 300, criterion = 'entropy')
model.fit(sex.reshape(-1, 1),  y.reshape(-1, 1))

y_pred = model.predict(test_x.Age.values.reshape(-1, 1))
y_pred
df = pd.DataFrame( y_pred, columns = ["Survived"])
df.shape
mid1 = pd.concat([test_x, df], axis = 1)

final = mid1.drop(columns= "Sex")
final.to_csv('submission.csv', index=False)