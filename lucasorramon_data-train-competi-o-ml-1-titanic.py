import pandas as pd
import numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
test.head()
train_title = train.copy()
test_title = test.copy()
# Extrai apenas os pronomes e coloquei em uma nova coluna chamada Title.
train_title['Title'] = train_title['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_title['Title'] = test_title['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_title.head()
train_title['Title'].value_counts()
title_mapped = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 
                'Master': 3, 'Dr': 3, 'Rev': 3, 'Mlle': 3, 'Ms': 3, 'Capt': 3, 
                'Col': 3, 'Countess': 3, 'Jonkheer': 3, 'Sir': 3, 'Mme': 3}
train_title['Title'] = train_title['Title'].map(title_mapped)
train_title['Title'].unique()
test_title['Title'] = test_title['Title'].map(title_mapped)
test_title['Title'].unique()
# Retirei os valores NAN do dataframe de testes
test_title.update(test_title['Title'].fillna(3))
test_title['Title'].isnull().sum()
train_title.drop(['Name'], inplace=True, axis=1)
test_title.drop(['Name'], inplace=True, axis=1)
train_title.head()
train_ticket = train_title.copy()
test_ticket = test_title.copy()
train_ticket.groupby(['Ticket']).count().sort_values('Survived', ascending=False).head(20)
# Na coluna Ticket tem um registro sem número, substituí por '0' para que eu pudesse tratar depois retirando
# as palavras e deixando os números.
train_ticket = train_ticket.replace('LINE', '0')
test_ticket = test_ticket.replace('LINE', '0')
split_train_ticket = train_ticket['Ticket'].str.split()
split_train_ticket = np.array(split_train_ticket.values)

split_test_ticket = test_ticket['Ticket'].str.split()
split_test_ticket = np.array(split_test_ticket.values)
# Retiro as palavras e salvo apenas os números dos tickets
new_ticket_train = []
for i in split_train_ticket:
    if np.shape(i) == (1,):
        new_ticket_train.append(i[0])
    else:
        new_ticket_train.append(i[-1])

new_ticket_test = []
for i in split_test_ticket:
    if np.shape(i) == (1,):
        new_ticket_test.append(i[0])
    else:
        new_ticket_test.append(i[-1])
new_ticket_train[0:5]
new_ticket_test[0:5]
new_ticket_train = np.array(new_ticket_train, dtype='int')
new_ticket_test = np.array(new_ticket_test, dtype='int')
train_ticket['Ticket'] = new_ticket_train

test_ticket['Ticket'] = new_ticket_test
train_ticket.head()
test_ticket.head()
train_ticket.isnull().sum()
test_ticket.isnull().sum()
train_cabin = train_ticket.copy()
test_cabin = test_ticket.copy()
train_cabin.drop(['Cabin'], axis=1, inplace=True)
test_cabin.drop(['Cabin'], axis=1, inplace=True)
train_embarked_sex = train_cabin.copy()
test_embarked_sex = test_cabin.copy()
train_embarked_sex.groupby(['Survived', 'Embarked']).count()
train_embarked_sex.head()
train_embarked_sex['Embarked'].unique()
train_embarked_sex.loc[train_embarked_sex['Embarked'].isnull(), 'Embarked'] = train_embarked_sex.mode()['Embarked'][0]
train_embarked_sex.isnull().sum()
test_embarked_sex.isnull().sum()
train_embarked_sex = pd.get_dummies(train_embarked_sex)
test_embarked_sex = pd.get_dummies(test_embarked_sex)
train_embarked_sex.head()
test_embarked_sex.head()
train_embarked_sex.corr()
train_embarked_sex.drop(['Sex_male', 'Embarked_Q'], axis=1, inplace=True)
test_embarked_sex.drop(['Sex_male', 'Embarked_Q'], axis=1, inplace=True)
train_embarked_sex.head()
test_embarked_sex.head()
train_age = train_embarked_sex.copy()
test_age = test_embarked_sex.copy()
train_age.loc[train_age['Age'].isnull(), 'Age'] = train_age.mean()['Age']

test_age.loc[test_age['Age'].isnull(), 'Age'] = test_age.mean()['Age']
train_age.head()
test_age.head()
train_age.corr()['Survived']
train_mean_std_max = train_age.copy()
test_mean_std_max = test_age.copy()
colunas = ['Pclass', 'Fare', 'Title', 'Sex_female', 'Embarked_C', 'Embarked_S']
train_mean_std_max['Mean'] = train_mean_std_max[colunas].mean(axis=1).values
train_mean_std_max['Std'] = train_mean_std_max[colunas].std(axis=1).values
train_mean_std_max['Max'] = train_mean_std_max[colunas].max(axis=1).values

test_mean_std_max['Mean'] = test_mean_std_max[colunas].mean(axis=1).values
test_mean_std_max['Std'] = test_mean_std_max[colunas].std(axis=1).values
test_mean_std_max['Max'] = test_mean_std_max[colunas].max(axis=1).values
train_mean_std_max.corr()
features = ['Pclass', 'Fare', 'Title', 'Sex_female', 'Embarked_C', 'Embarked_S', 'Mean', 'Std', 'Max']
X = train_mean_std_max[features]
Y = train_mean_std_max['Survived']
X
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
model_forest = RandomForestClassifier(random_state=1)
model_forest.fit(x_train, y_train)
y_pred_train = model_forest.predict(x_train)
accuracy_score(y_train, y_pred_train)
y_pred_train = model_forest.predict(x_test)
y_pred_train_proba = model_forest.predict_proba(x_test)
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

%matplotlib inline
rcParams['figure.figsize'] = 8, 4
cm = confusion_matrix(y_test, y_pred_train)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Matriz de Confusão')
plt.ylabel('True label')
plt.xlabel('Predicted label')
accuracy_score(y_test, y_pred_train)
print(classification_report(y_test, y_pred_train))
fp, tp, thresholds = roc_curve(y_test, y_pred_train_proba[:, 1])
plt.plot(fp, tp)

plt.plot([0, 1], [0, 1], '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC')
auc(fp, tp)
model_forest.fit(X, Y)
end_test = test_mean_std_max[features]
y_pred_test_proba = model_forest.predict_proba(end_test)
y_pred_test_proba = y_pred_test_proba[:, 1]
passageiro_id = test['PassengerId']
resultado = pd.concat([passageiro_id, pd.DataFrame(y_pred_test_proba, columns=['Survived'])], axis=1)
resultado.to_csv('submission.csv', index=False)
