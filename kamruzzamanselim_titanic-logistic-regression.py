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
train_data['survived'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
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
sns.countplot(x = 'survived', data = train_data, palette = 'hls')
plt.show()
count_no_sur = len(train_data[train_data['survived']==0])
count_sur = len(train_data[train_data['survived']==1])
pct_of_no_sur = count_no_sur/(count_no_sur+count_sur)
print("percentage of not survived is", pct_of_no_sur*100)
pct_of_sub = count_sur/(count_no_sur+count_sur)
print("percentage of survived", pct_of_sub*100)
train_data.groupby('survived').mean()

# , 'pclass', 'age', 'sibsp', 'sex_male', 'embarked_S'
X = train_data.drop(['survived','passengerid'], axis=1)
y = train_data['survived']
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
cols = ['cabin', 'sibsp', 'age', 'pclass']
X=X[cols]
# pd.concat([X,y], axis=1).corr()
# X
# pd.concat([X,y])
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)