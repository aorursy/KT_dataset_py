# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

processed_data = pd.concat([train.drop('Survived', axis = 1), test])
def fill_age(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age
    
processed_data['Age'] = processed_data[['Age','Pclass']].apply(fill_age, axis = 1)
import re
processed_data['Title'] = processed_data['Name'].apply(lambda x: re.split(',|\.',x)[1].lstrip())
titles = pd.get_dummies(processed_data['Title'], drop_first=True)
embark = pd.get_dummies(processed_data['Embarked'], drop_first=True)
sex = pd.get_dummies(processed_data['Sex'], drop_first=True)
pclass = pd.get_dummies(processed_data['Pclass'], drop_first=True)

processed_data.drop('Cabin', axis = 1, inplace = True)
processed_data.at[processed_data['Fare'].isna(),'Fare'] = processed_data[processed_data['Pclass'] == 3]['Fare'].mean()

processed_data = pd.concat([processed_data,embark,sex,pclass,titles], axis = 1)
processed_data.drop(['Pclass','Embarked','Sex','Name','Ticket','Title'], axis =1, inplace = True)
processed_train = processed_data.iloc[0:891,:]
processed_test = processed_data.iloc[891:,:]
y_train = train['Survived']
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(processed_train,y_train)
#pd.DataFrame(data = lr.coef_, columns=processed_data.columns)
predictions = pd.Series(lr.predict(processed_test))
second_submission = pd.concat([processed_test['PassengerId'], predictions], axis = 1)
second_submission.columns = ['PassengerId', 'Survived']
second_submission.to_csv('Second_Submission.csv', index = False)