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
from sklearn.preprocessing import Imputer

from sklearn.linear_model import LogisticRegression

import pandas as pd 

from sklearn.cross_validation import train_test_split





df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

my_imputer = Imputer()

lr = LogisticRegression()



sex = df['Sex']

sex_test = df_test['Sex']

sex.replace(to_replace = 'male', value = 0, inplace = True)              #df['Sex'].replace({

                                                                         #'male':0,'female':1

                                                                         #   },inplace=True

sex.replace(to_replace = 'female', value = 1, inplace = True)

sex_test.replace(to_replace = 'male', value = 0, inplace = True)              #df['Sex'].replace({

                                                                         #'male':0,'female':1

                                                                         #   },inplace=True

sex_test.replace(to_replace = 'female', value = 1, inplace = True)





Xtrain = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

Xtest = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

Xtrain=my_imputer.fit_transform(Xtrain)

Xtest=my_imputer.fit_transform(Xtest)

ytrain = df['Survived']



lr.fit(Xtrain, ytrain)

ytest = lr.predict(Xtest)



submission = pd.DataFrame({"PassengerId": df_test['PassengerId'], "Survived": ytest})

submission.to_csv('titanic2.csv', index=False)
