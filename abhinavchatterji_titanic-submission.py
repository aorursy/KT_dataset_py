
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection  import train_test_split

import os
print(os.listdir("../input"))
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_train.head(5)
titanic_train.describe()
titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
titanic_train.isnull().sum()
sb.heatmap(titanic_train.corr())
average_Pclass1_age = titanic_train[titanic_train['Pclass'] == 1].Age.mean()
average_Pclass2_age = titanic_train[titanic_train['Pclass'] == 2].Age.mean()
average_Pclass3_age = titanic_train[titanic_train['Pclass'] == 3].Age.mean()
print("Average Pclass = 1 age:", average_Pclass1_age )
print("Average Pclass = 2 age:", average_Pclass2_age )
print("Average Pclass = 3 age:", average_Pclass3_age )
titanic_train.loc[titanic_train['Pclass'] == 1, 'Age'] = titanic_train.loc[titanic_train['Pclass'] == 1, 'Age'].fillna(value = average_Pclass1_age)
titanic_train.loc[titanic_train['Pclass'] == 2, 'Age'] = titanic_train.loc[titanic_train['Pclass'] == 2, 'Age'].fillna(value = average_Pclass2_age)
titanic_train.loc[titanic_train['Pclass'] == 3, 'Age'] = titanic_train.loc[titanic_train['Pclass'] == 3, 'Age'].fillna(value = average_Pclass3_age)
titanic_train.head(5)
plt.scatter(titanic_train.Pclass, titanic_train.Age)
fig, axs = plt.subplots(1,3)
titanic_train.loc[titanic_train['Pclass'] == 1, 'Fare'].hist(ax = axs[0])
titanic_train.loc[titanic_train['Pclass'] == 2, 'Fare'].hist(ax = axs[1])
titanic_train.loc[titanic_train['Pclass'] == 3, 'Fare'].hist(ax = axs[2])
plt.tight_layout()
titanic_train.drop(['Pclass', 'Fare'], axis = 1, inplace = True)
titanic_train.head(5)
Sex_is_male = pd.get_dummies(titanic_train['Sex'],drop_first=True)
titanic_train_dummy = pd.concat([titanic_train, Sex_is_male], axis = 1)
titanic_train_dummy.drop(['Sex'], axis = 1, inplace = True)
titanic_train_dummy.head(5)
X = titanic_train_dummy[['Age', 'SibSp', 'Parch', 'male']].values
Y = titanic_train_dummy[['Survived']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3, random_state=25)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
titanic_test_copy = titanic_test.copy()
average_Pclass1_age_test = titanic_test[titanic_test['Pclass'] == 1].Age.mean()
average_Pclass2_age_test = titanic_test[titanic_test['Pclass'] == 2].Age.mean()
average_Pclass3_age_test = titanic_test[titanic_test['Pclass'] == 3].Age.mean()
titanic_test.loc[titanic_test['Pclass'] == 1, 'Age'] = titanic_test.loc[titanic_test['Pclass'] == 1, 'Age'].fillna(value = average_Pclass1_age_test)
titanic_test.loc[titanic_test['Pclass'] == 2, 'Age'] = titanic_test.loc[titanic_test['Pclass'] == 2, 'Age'].fillna(value = average_Pclass2_age_test)
titanic_test.loc[titanic_test['Pclass'] == 3, 'Age'] = titanic_test.loc[titanic_test['Pclass'] == 3, 'Age'].fillna(value = average_Pclass3_age_test)

titanic_test.drop(['PassengerId', 'Pclass', 'Name', 'Ticket','Fare','Cabin', 'Embarked'], axis = 1, inplace = True)

Sex_is_male_test = pd.get_dummies(titanic_test['Sex'],drop_first=True)
titanic_test_dummy = pd.concat([titanic_test, Sex_is_male_test], axis = 1)
titanic_test_dummy.drop(['Sex'], axis = 1, inplace = True)
titanic_test_dummy.head(5)
X_test_final = titanic_test_dummy[['Age', 'SibSp', 'Parch', 'male']].values
Y_pred = logreg.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
cm
df_cm = pd.DataFrame(cm, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sb.heatmap(df_cm, annot=True)
Y_pred_final = logreg.predict(X_test_final)
Survival = pd.concat([titanic_test_copy, pd.DataFrame(Y_pred_final, columns = ["Survived"])], axis = 1)
Result_df = Survival[['PassengerId','Survived']]
os.chdir("/kaggle/working/")
filename = 'Titanic Prediction.csv'

Result_df.to_csv(filename,index=False)

print('Saved file: ' + filename)
print(os.listdir("../working"))





