# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib as plt
!pwd
df=pd.read_csv("/kaggle/input/titanic/train.csv")
df
y=df.drop('Survived',axis=1)

df.isnull().sum()
df=df.drop('Age',axis=1)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)
df=df.drop('Embarked',axis=1)
df=df.drop('Fare',axis=1)
y=df.Survived
y
df.Sex=df.Sex.apply(['male','female'].index)
X=df.iloc[:,2:]
X
df_test=pd.read_csv("/kaggle/input/titanic/test.csv")

df_test=df_test.drop('Name',axis=1)
df_test=df_test.drop('Age',axis=1)
df_test=df_test.drop('Ticket',axis=1)
df_test=df_test.drop('Cabin',axis=1)
df_test=df_test.drop('Embarked',axis=1)
df_test=df_test.drop('Fare',axis=1)
df_test.Sex=df_test.Sex.apply(['male','female'].index)
df_test
y_test=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
y_test
x_test=df_test.iloc[:,1:]
x_test
from sklearn import tree
model=tree.DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=4)
model=model.fit(X,y)
df_test.isnull().sum()

y_pred=model.predict(x_test)
