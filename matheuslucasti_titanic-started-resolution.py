import pandas as pd
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#tirando os dados que não serão utilizados
train.drop(['Name','Ticket','Cabin'], axis=1, inplace = True)
test.drop(['Name','Ticket','Cabin'], axis=1, inplace = True)

# print(train.head())

#trocar texto por numero
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

# new_data_train.isnull().sum().sort_values(ascending=False).head(10) verificar se tem coluna null
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

# new_data_test.isnull().sum().sort_values(ascending=False).head(10) verificar se tem coluna null
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

X = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']

tree = DecisionTreeClassifier(max_depth=7,random_state=1)

tree.fit(X,y)

print(tree.score(X,y))
# new_data_train.head()

# exportando o arquivo com os dados
submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)

submission.to_csv('submission.csv', index= False)
