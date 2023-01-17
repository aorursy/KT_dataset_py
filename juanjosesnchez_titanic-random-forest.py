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

import pandas as pd

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder 





# Load training set 

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
## Obtenemos la columna de supervivencia

y = train_data['Survived']

features = ["Pclass", "Sex", "SibSp", "Parch", "Name","Fare","Embarked","Age"]

all_other = train_data[features]

X = all_other.copy()

X.head()
#Data analysis 

sns.heatmap(X.isnull(), cbar=False)



print("Age has null values")



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
# Feature enginering 



#Extracting title 



le = LabelEncoder() 



    

def transformation(data):

    data['title'] = data['Name'].apply(lambda x : x.split(',')[1].strip().split('.')[0].strip())

    data['Sex'] = (data['Sex'] == 'male') * 1.0

    data['title']= le.fit_transform(data['title']) 

    data['SibSp'] = data['SibSp'].apply(lambda x : 1 if x > 0 else 1)

    data = data.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

    data = data.drop(['Name'], axis=1)

    data["safeAge"] = data['Age'].apply(lambda x : 1 if 29 <= x <= 57 else 0)

    data = data.fillna(data.mean())

    return data





X = transformation(X)

sns.heatmap(X.isnull(), cbar=False)

X.head()
# entrenar el random forest



model = RandomForestClassifier(n_estimators=1000, 

                                max_depth=None,

                                min_samples_split=2, 

                                random_state=81).fit(X, y)



# se obtienen las importancias de las cols (Impurities)

importances = model.feature_importances_

# se ordenan los indices

indices = np.argsort(importances)[::-1]

# se ordena el dataset

cols = X.columns[indices]



print(importances)

print(indices)

print(cols)



print("Cross-Validation Score:",cross_val_score(model, X, y, cv=5).mean())  
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_all_other = test_data[features]

X_test = test_all_other.copy()

X_test = transformation(X_test)



X_test.head()
predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")