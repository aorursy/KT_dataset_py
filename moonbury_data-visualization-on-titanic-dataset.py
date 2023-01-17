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
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



pd.set_option('precision',4)

pd.set_option('display.width',120)



titanic_df = pd.read_csv("../input/train.csv")



del titanic_df['Name']

#del titanic_df['PassengerId']

del titanic_df['Ticket']

del titanic_df['Cabin']

#titanic_df['Embarked_int'] = pd.Series([titanic_df['Embarked']=='S')





numsex = {"male":1 ,"female" :2}

titanic_df['Sex'] = titanic_df['Sex'].replace(numsex)

titanic_df['Sex'] = titanic_df['Sex'].convert_objects(convert_numeric=True)





numembark = {"S":1 ,"C" :2, "Q":3}

titanic_df['Embarked'] = titanic_df['Embarked'].replace(numembark)

titanic_df['Embarked'] = titanic_df['Embarked'].convert_objects(convert_numeric=True)

titanic_df['Embarked'].fillna(1)



titanic_df['Embarked'].fillna(titanic_df['Age'].mean())



titanic_df.head()
print(titanic_df.describe())
boxplots = titanic_df.boxplot(return_type='axes')
boxplots = titanic_df.boxplot(column='Age',by='Survived',return_type='axes')