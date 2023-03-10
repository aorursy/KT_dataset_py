# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
print('Train columns with null values:\n', train_data.isnull().sum())

print("-"*20)

#C?? nhi???u null trong Age v?? Cabin
#Fill gi?? tr??? trung binh cho Age

train_data.Age.fillna(train_data.Age.mean(),inplace=True)
#Cabin l?? thu???c t??nh ?????nh danh v?? qu?? nhi???u null ==> kh??ng gi??p ??t cho model --> x??a

#S??? v?? l?? bi???n ?????nh danh ng???u nhi??n, trong tr?????ng h???p n??y gi?? v?? kh??ng gi??p it cho vi???c ph??n t??ch --> x??a

#PassengerId bi???n ?????nh danh ng???u nhi??n, kh??ng c?? ??ch cho vi???c ph??n t??ch --> X??a

train_data.drop(['Cabin','Ticket','PassengerId'], axis=1, inplace = True)
# Embarked c?? 2 gi?? tr??? NaN, fill b???ng gi?? tr??? ?????nh danh xu???t hi???n nhi???u nh???t

train_data.fillna(train_data.Embarked.mode()[0],inplace=True)
#Famile

train_data['nMembers_Family']=train_data['SibSp']+train_data['Parch'] + 1



sns.countplot(x='nMembers_Family', data=train_data, hue='Survived')



train_data.drop(['SibSp','Parch'],axis=1, inplace = True)



train_data['IsAlone'] = 1 #khoi tao la 1

train_data['IsAlone'].loc[train_data['nMembers_Family'] > 1] = 0 # c???p nh???t l???i t??nh tr???ng

train_data
sns.countplot(x='Pclass', data=train_data, hue='Survived');
print('Train columns with null values:\n', train_data.isnull().sum())
from sklearn import svm,tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process





y = train_data["Survived"]

print(y.shape)

features = ["Pclass", "Sex", "SibSp", "Parch","Age"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

print(X)

model = svm.SVC(kernel = 'linear')

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission1.csv', index=False)

print("Your submission was successfully saved!")