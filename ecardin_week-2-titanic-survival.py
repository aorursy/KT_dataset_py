import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
fix, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(df_train.isnull())
plt.show()
# Age, Cabin, Embarked has at least null
categorical = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
nrow = 2; ncol = 3
fig, axs = plt.subplots(nrow, ncol, figsize=(nrow*5, ncol*2))
i = 0
for r in range(0,nrow):
    for c in range(0,ncol):  
        ax = axs[r][c]
        sns.countplot(df_train[categorical[i]], hue=df_train["Survived"], ax=ax)
        ax.set_title(categorical[i])
        i += 1
        ax.legend() 
        
plt.tight_layout() 
sns.boxplot(x='Embarked', y='Age', data=df_train)
plt.title("box plot")
plt.show()
def extract_name_info(name):
    if 'Mr.' in name:
        return 1
    elif 'Mrs.' in name:
        return 2
    elif 'Miss.' in name:
        return 3
    else:
        return 0

# quantify data roughly
def quantify(df_train):
    df_train_copy = df_train.copy()
    df_train_copy['Name'] = df_train['Name'].apply(extract_name_info)
    df_train_copy['Sex'] = df_train['Sex'].apply(lambda v: int(v == 'male'))
    df_train_copy = df_train_copy.drop(['Ticket'], axis=1)
    df_train_copy = df_train_copy.drop(['PassengerId'], axis=1)
    df_train_copy['Cabin'] = df_train_copy['Cabin'].fillna('X').apply(lambda v: v[0])
    df_train_copy['Age'] = df_train_copy['Age'].fillna(df_train_copy['Age'].mean())
    df_train_copy['Cabin'] = df_train_copy['Cabin'].apply(lambda v: ord(v))
    df_train_copy['Embarked'] = df_train_copy['Embarked'].fillna('S').apply(lambda v: ord(v))
    return df_train_copy
df_train_quantify = quantify(df_train)
df_test_quantify = quantify(df_test)
df_train_quantify.head()
X = df_train_quantify.drop(['Survived'], axis=1)
y = df_train_quantify['Survived']
clf = XGBClassifier()
clf.fit(X, y)
plot_importance(clf)
plt.show()
df_test_quantify = quantify(df_test)
y_pred = clf.predict(df_test_quantify)
df_test.head()
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()