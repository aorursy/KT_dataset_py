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
from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import roc_auc_score
X=pd.read_csv('../input/titanic/train.csv')
Y=X.pop('Survived')
X.describe() #I see im missing some data in the Age, so im gonna fix that
X['Age'].fillna(X.Age.mean(), inplace=True)

X.describe() #Now I filled the missing numbers with the mean, in Age. Meaning I now have the same amount of values in all the colums
numeric_variables= list(X.dtypes[X.dtypes != 'object'].index)

X[numeric_variables].head()

#To make it easier Im just focusing on numeric values, for now. 
model= RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)



model.fit(X[numeric_variables], Y)
Y_oob = model.oob_prediction_



print("c-stat:", roc_auc_score(Y, Y_oob))
X.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True) #here I drop the values in Name, Ticket and PassengerId, 

#I do that because these columns are very uniq and therefor wont help my predictions much
def clean_cabin(x):

    try:

        return x[0]

    except TypeError:

        return "None"

X["Cabin"]= X.Cabin.apply(clean_cabin)

#Here I clean the numbers from the Cabin, so its only cabin B,E,C ect. and not B2 B22. That makes it less uniq.

#I want it to be less uniq to help my predictions more.
categorical_variables = ["Sex","Cabin","Embarked"]



for variable in categorical_variables: 

    X[variable].fillna("Missing", inplace=True)

    dummies= pd.get_dummies(X[variable], prefix=variable)

    

    X= pd.concat([X, dummies], axis=1)

    X.drop([variable], axis=1, inplace=True)

    

#Here I exchange the NaN values with missing, and then I drop the NaN
X #just to see what my dataset looks like at this point
model = RandomForestRegressor(100, oob_score=True,n_jobs=-1, random_state=42)

model.fit(X,Y)

print ("C-stat: ", roc_auc_score(Y,model.oob_prediction_))
