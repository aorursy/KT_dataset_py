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
#Import Dataset
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_list = [df_train,df_test]
df = pd.concat(df_list)
df.head()
df.columns = map(str.lower, df.columns)

df['name'] = df['name'].str.lower()
df['sex'] = df['sex'].str.lower()
df['age'] = df['age'].replace(to_replace = np.nan, value = 30)
# df[0:890].describe()
df['fare'] = df['fare'].replace(to_replace = np.nan, value = 32)
df = df.drop('name', axis = 1)
df.describe()
features = ["sex",'embarked']
a = pd.get_dummies(df[features])
df = pd.concat([df,a], axis = 1)
df['cabin'] = df['cabin'].fillna(0)
df['cabin'] = df['cabin'].astype(bool).astype(int)
df1 = df.drop(['ticket','sex', 'embarked'], axis = 1)
df1
train_data = df1.iloc[0:891]
test_data = df1.iloc[891:1309]
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
train_data
# , 'pclass', 'age', 'sibsp', 'sex_male', 'embarked_S'
X = train_data.drop(['survived','passengerid', 'pclass', 'age', 'sibsp', 'sex_male', 'embarked_S'], axis=1)
y = train_data['survived']
# pd.concat([X,y], axis=1).corr()
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
y_pred
output = pd.DataFrame({'Passengerid': test_data.passengerid, 'Survived': y_pred})
output.to_csv('svm.csv', index=False)
print("Your submission was successfully saved!")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
