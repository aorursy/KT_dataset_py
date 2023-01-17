import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train= pd.read_csv("/kaggle/input/titanic/train.csv")
titanic_train.head(10)
titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_test.head()
titanic_train.shape
titanic_train.describe()
titanic_train.Survived.value_counts()
survived = titanic_train.Survived
sns.countplot(survived)
plt.show()
sex_women = titanic_train.loc[titanic_train.Sex == 'female']["Survived"]
alive_women = sum(sex_women)
num_women = len(sex_women)
print(f'number of alive women: {alive_women}')
print(f'total number of women: {num_women}')
sex_men = titanic_train.loc[titanic_train.Sex == 'male']["Survived"]
alive_men = sum(sex_men)
num_men = len(sex_men)
print(f'number of alive men: {alive_men}')
print(f'total number of men: {num_men}')
rate_survived_men = alive_men / num_men
rate_survived_women = alive_women / num_women
print(f'rate of survived men: {rate_survived_men}')
print(f'rate of survived women: {rate_survived_women}')
class_alive_men = titanic_train.loc[(titanic_train.Sex == 'male') & (titanic_train.Survived == 1)]['Pclass']
class_dead_men = titanic_train.loc[(titanic_train.Sex == 'male') & (titanic_train.Survived == 0)]['Pclass']
sns.countplot(class_alive_men, label = 'alive men')
plt.title('alive men')
plt.show()
sns.countplot(class_dead_men, label = 'dead men')
plt.title('dead men')
plt.show()
class_alive_women = titanic_train.loc[(titanic_train.Sex == 'female') & (titanic_train.Survived == 1)]['Pclass']
class_dead_women = titanic_train.loc[(titanic_train.Sex == 'female') & (titanic_train.Survived == 0)]['Pclass']
sns.countplot(class_alive_women)
plt.title('alive women')
plt.show()
sns.countplot(class_dead_women)
plt.title('dead women')
plt.show()
age = pd.cut(titanic_train['Age'], [0, 18, 80])
titanic_train.pivot_table('Survived', ['Sex', age], 'Pclass')
titanic_train.isna().sum()
y = titanic_train["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(titanic_train[features])
X_testt = pd.get_dummies(titanic_test[features])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
model = neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')
from sklearn.metrics import accuracy_score
print(accuracy_score( y_test, y_pred))
print(np.mean(y_test==y_pred))
predictions = neigh.predict(X_testt)
output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")