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
import matplotlib.pyplot as plt #Visualization

from sklearn.neighbors import KNeighborsClassifier #knn

from sklearn.svm import SVC #Support_Vector_Machine 

from sklearn.linear_model import LogisticRegression #Logistic_Regression 

from sklearn.ensemble import RandomForestClassifier #RandomForest 

from sklearn.preprocessing import LabelEncoder
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')





df_train.head(10)
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



df_test.head(10)
print('shape of the training data {}'.format(df_train.shape))

print('The number of passengers who survived {}'.format(np.array(df_train['Survived'].value_counts()[1])))

print('The number of passengers who didn\'t survived {}'.format(np.array(df_train['Survived'].value_counts()[1])))

print('There were {} men while {} women on Titanic'.format(df_train['Sex'].value_counts()[0], df_train['Sex'].value_counts()[1]))

print('Average age of passengers was {}'.format(df_train['Age'].mean()))

print('Average age of female passengers was {} while that of men was {}'.format(df_train.groupby('Sex')['Age'].mean()[0],\

                                                                               df_train.groupby('Sex')['Age'].mean()[1]))



print()

print()



print("Checking for NULL values in each column")



print(df_train.isnull().sum())



df_train.groupby('Embarked')['Survived'].value_counts()



y = df_train['Survived'].value_counts()

x= ['Died', 'Survived']



plt.bar(x,y, color='#df94f2')

plt.title('Comparision of number of passengers who survived and those who couldn\'t')

plt.tight_layout()

plt.show()
plt.hist(df_train['Age'], alpha=0.7, edgecolor= '#0d592f', color='#72e8a7')

plt.axvline(df_train['Age'].mean(), color='k')

plt.xlabel('Age of passengers')

plt.title('Distribution of ages of passengers')

plt.tight_layout()

plt.show()
x = np.array(df_train['Embarked'].value_counts())

y = np.array(df_train['Embarked'].value_counts().index)

colors = ['#669cd1', '#dd7ee6' , '#d9f479']

plt.pie(x, labels=y, shadow=True, autopct = '%1.1f%%', colors = colors)



plt.show()
df_train['Age'].fillna(value= df_train['Age'].mean(), inplace=True)

df_train['Embarked'].fillna(method='ffill', inplace=True)



#df_train.isnull().sum()



df_test['Age'].fillna(value= df_test['Age'].mean(), inplace=True)

df_test['Fare'].fillna(value= df_test['Fare'].mean(), inplace=True)



#df_test.isnull().sum()



sex = LabelEncoder()

embark = LabelEncoder()



df_train['Sex'] = sex.fit_transform(df_train['Sex'])

df_test['Sex'] = sex.fit_transform(df_test['Sex'])



df_train['Embarked'] = embark.fit_transform(df_train['Embarked'])

df_test['Embarked'] = embark.fit_transform(df_test['Embarked'])



features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



X_train = df_train[features]

y_train = df_train['Survived']



X_test = df_test[features]
rfc = RandomForestClassifier(max_depth = 8 , n_estimators = 125, random_state= 4)

#rfc = KNeighborsClassifier(3)



rfc = rfc.fit(X_train, y_train)

predict =  rfc.predict(X_test)



output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predict})

output.to_csv('my_submission.csv', index=False)