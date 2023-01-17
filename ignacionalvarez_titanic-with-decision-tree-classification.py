import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
example_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

SEED = 42
TARGET = 'Survived'
df_train.head()
df_train.info()
df_train.describe()
df_train['Sex'].value_counts()
df_train['Cabin'].value_counts()
df_train['Embarked'].value_counts()
df_train['Parch'].value_counts()
df_train.isnull().sum()
def is_numerical(col):
    return (col.dtype == 'float64') or (col.dtype == 'int64') or (col.dtype == 'int32')

def is_categorical(col):
    return (col.dtype == 'category')
def drop_columns(data, drop_cols):
    for col in data.columns:
        if col in drop_cols:
            data.drop(col, axis=1, inplace = True)
def become_categorical(data, categorical_cols):
     for col in data.columns:
        if col in categorical_cols:
            data[col] = data[col].astype('category').cat.codes
def change_nulls(data, null_cols):
    for col in data.columns:
        if col in null_cols: 
            if is_numerical(data[col]):
                data[col].fillna(data[col].median(), inplace=True)
            elif is_categorical(data[col]):
                data[col].fillna(data[col].mode().iloc[0], inplace=True)
def remove_outliers(data, columns_outliers):
    for col in data.columns:
        if col in columns_outliers: 
            data = data.loc[data[col] > (data[col].mean() - data[col].std() * 3)]
            data = data.loc[data[col] < (data[col].mean() + data[col].std() * 3)]
    return data

drop_cols = ['Name', 'Ticket', 'PassengerId', 'Cabin']
drop_columns(data=df_train, drop_cols=drop_cols)
categorical_cols = ['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']
become_categorical(data=df_train, categorical_cols=categorical_cols)
null_cols = ['Age', 'Fare']
change_nulls(data=df_train, null_cols=null_cols)
columns_outliers = ['Fare', 'Age']
df_train = remove_outliers(data=df_train, columns_outliers=columns_outliers)
fig,  (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20,5))

sns.distplot(df_train.Fare, ax=ax1)
sns.boxplot(x=TARGET, y='Fare', data=df_train, ax=ax2)
sns.swarmplot(x=TARGET, y='Fare', data=df_train, ax=ax3)
sns.violinplot(x=TARGET, y='Fare', data=df_train, ax=ax4)
fig,  (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20,5))

sns.distplot(df_train.Age, ax=ax1)
sns.boxplot(x=TARGET, y='Age', data=df_train, ax=ax2)
sns.swarmplot(x=TARGET, y='Age', data=df_train, ax=ax3)
sns.violinplot(x=TARGET, y='Age', data=df_train, ax=ax4)
df_train.shape
y = df_train[TARGET]
X = df_train.drop(TARGET, axis=1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)
model = DecisionTreeClassifier(min_samples_leaf = 2,max_depth=5, random_state=SEED)
model.fit(X_train, y_train)
plot_tree(model) 

yhat_test = model.predict(X_test)
score = accuracy_score(y_test, yhat_test)
score
df_test.head()
df_test_id = df_test['PassengerId']
df_test.drop('PassengerId', axis=1, inplace=True)
drop_cols = ['Name', 'Ticket', 'PassengerId', 'Cabin']
drop_columns(data=df_test, drop_cols=drop_cols)
categorical_cols = ['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch']
become_categorical(data=df_test, categorical_cols=categorical_cols)
null_cols = ['Age', 'Fare']
change_nulls(data=df_test, null_cols=null_cols)
df_test.head()
predictions = model.predict(df_test)
predictions
example_submission.head()
a = pd.Series(predictions)
df_lp = pd.DataFrame([df_test_id,a]).T
df_lp.columns = ['PassengerId', 'Survived']
df_lp.head()
df_lp.to_csv('submission.csv', index=False)