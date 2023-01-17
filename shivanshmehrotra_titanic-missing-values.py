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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
features = ["Pclass","Sex","SibSp","Parch"]

X = pd.get_dummies(train_data[features])

y=(train_data["Survived"])

X_test = pd.get_dummies(test_data[features])
imputer = SimpleImputer(missing_values = np.nan,  

                        strategy ='mean') 

steps = [('scaler',StandardScaler()),(('imputation',imputer)),('logreg',LogisticRegression())]

pipeline = Pipeline(steps)

pipeline.fit(X,y)
y_pred = pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Saved!")

output.shape