import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import tree

import graphviz

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# percent of null value

def perisnull(df):

    for data in df.columns:

        isnulls = df[data].isnull().sum()/len(df)*100

        print('Percent missing value of', data,' : ', '%.2f' %isnulls)
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
print(df_train.info())

print('-'*80)

print(df_test.info())
df_train.columns
perisnull(df_train)
perisnull(df_test)
# drop Cabin columns in train and test dataset

df_train.drop(['Cabin'], axis=1, inplace = True)

df_test.drop(['Cabin'], axis=1, inplace = True)
df_train.drop(['PassengerId', 'Name','Ticket'], axis=1, inplace = True)

df_test.drop(['PassengerId', 'Name','Ticket'], axis=1, inplace = True)
perisnull(df_train)
perisnull(df_test)
plt.hist(df_train['Age'],bins=100, histtype='bar')

plt.show()
plt.hist(df_test['Age'],bins=100, histtype='bar')

plt.show()
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_train['Age'].median())
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].hist(df_train['Fare'],bins=100, histtype='bar')

axs[0].set_title('Train data_Fare')

axs[1].hist(df_test['Fare'],bins=100, histtype='bar')

axs[1].set_title('Test data_Fare')

plt.show()
df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].mode()[0])
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
perisnull(df_train)
perisnull(df_test)
le = preprocessing.LabelEncoder()

df_train['Sex'] = le.fit_transform(df_train['Sex'])

df_train.head()
df_test['Sex'] = le.fit_transform(df_test['Sex'])

df_test.head()
df_train = pd.get_dummies(df_train, columns = ['Embarked'])

df_test = pd.get_dummies(df_test, columns = ['Embarked'])

df_train = pd.get_dummies(df_train, columns = ['Pclass'])

df_test = pd.get_dummies(df_test, columns = ['Pclass'])
df_train.head()
df_train['Age_cut'] = pd.cut(df_train['Age'], bins=5)
df_train['Fare_cut'] = pd.cut(df_train['Fare'], bins=4)
df_train.head()
df_train.groupby(['Age_cut']).mean()
df_train.groupby(['Fare_cut']).mean()
df_test['Age_cut'] = pd.cut(df_test['Age'], bins=5)

df_test['Fare_cut'] = pd.cut(df_test['Fare'], bins=4)
df_train.drop(['Age', 'Fare'], axis =1, inplace=True)

df_test.drop(['Age', 'Fare'], axis =1, inplace=True)
print(df_train.columns)

print(df_test.columns)
df_train['Age_cut'] = df_train['Age_cut'].astype('category').cat.codes

df_train['Fare_cut'] = df_train['Fare_cut'].astype('category').cat.codes
df_train.groupby(['Age_cut']).mean()
df_train.groupby(['Fare_cut']).mean()
df_train.head()
df_test['Age_cut'] = df_test['Age_cut'].astype('category').cat.codes

df_test['Fare_cut'] = df_test['Fare_cut'].astype('category').cat.codes
X = df_train.drop(['Survived'], axis=1)

y = df_train['Survived']
X.head()
y.head()
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.info()
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train ,y_train)
dot_data = tree.export_graphviz(clf, out_file=None, max_depth=None,

                                filled=True, rounded=True,

                                special_characters=True,feature_names = X.columns,

                                class_names=['No','Yes']) 



graph = graphviz.Source(dot_data) 

graph.render("Titanic_full") 

graph 
y_pred = clf.predict(X_test)

print(y_pred)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

acc_decision_tree
y_eva = clf.predict(df_test)

print(y_eva)
# Generate Submission File 

submission = pd.DataFrame({

        "PassengerId": np.arange(892,1310,1),

        "Survived": y_eva

    })

submission.to_csv('../working/submission.csv', index=False)