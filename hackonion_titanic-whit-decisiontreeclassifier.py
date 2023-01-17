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

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/titanic/train.csv')

df_train.head()
df_test = pd.read_csv('../input/titanic/test.csv')

df_test.head()
df_train.columns
df_test.columns
df_train.info()
df_test.info()
sns.countplot(x=df_train.Survived,hue=df_train.Sex,data=df_train,palette='Set3_r')
sns.countplot(x=df_train.Survived,hue=df_train.Pclass,data=df_train,palette='Set3_r')
df_train.isnull().sum().sort_values(ascending=False)
miss_data = df_train.isnull()

sns.heatmap(miss_data,yticklabels=False,cbar=False,cmap='gray')

plt.title('Miss data in the training data')
df_test.isnull().sum().sort_values(ascending=False)
miss_data = df_test.isnull()

sns.heatmap(miss_data,yticklabels=False,cbar=False,cmap='gray')

plt.title('Miss data in the test data')
plt.figure(figsize=(30,10))

sns.heatmap(df_train.corr(),cmap='BuPu',annot=True)

plt.show()
corr = df_train.corr()

top_corr_features = corr.index[abs(corr['Survived'])>0.2]

top_corr_features
df_train = df_train[['PassengerId','Survived', 'Pclass', 'Sex','Age']]

df_test = df_test[[ 'PassengerId','Pclass', 'Sex','Age']]
mean_age = round(df_train.Age.mean())

df_train['Age'] = df_train["Age"].replace(np.nan, mean_age)

df_test['Age'] = df_test["Age"].replace(np.nan, mean_age)
le = preprocessing.LabelEncoder()

df_train = df_train.apply(le.fit_transform)



df_train.head()
le = preprocessing.LabelEncoder()

df_test = df_test.apply(le.fit_transform)

df_test.head()
X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['Survived','PassengerId'],axis=1), 

                                                    df_train['Survived'], test_size=0.1, 

                                                    random_state=101)
decisiontree = DecisionTreeClassifier()

decisiontree.fit(X_train, y_train)



pred = decisiontree.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, pred))

print('Score train', decisiontree.score(X_train,y_train))

print('Score test', decisiontree.score(X_test,y_test))
df = pd.read_csv('../input/titanic/test.csv')

Id = df['PassengerId']

predictions = decisiontree.predict(df_test.drop('PassengerId', axis=1))



result = pd.DataFrame({ 'PassengerId' : Id, 'Survived': predictions })

result.head()
result.to_csv('submission.csv' , index=False)