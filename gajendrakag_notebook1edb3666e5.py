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
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.head()
train_data['Cabin']=train_data['Cabin'].replace({np.NaN:'unknown'})
train_data['Age']=train_data['Age'].replace({np.NaN:np.mean(train_data['Age'])})
train_data
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
test_data['Cabin']=test_data['Cabin'].replace({np.NaN:'unknown'})
test_data['Age']=test_data['Age'].replace({np.NaN:np.mean(test_data['Age'])})
test_data
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch","Age","Cabin"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
X_final,X_test_final=X.align(X_test,'inner',axis=1)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_final, y)
predictions = model.predict(X_test_final)
cv_scores=cross_val_score(model,X_final,y)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
print(cv_scores,np.mean(cv_scores))
