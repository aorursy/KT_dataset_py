import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

np.random.seed(44)
df_train = pd.read_csv('dados/train.csv')
df_test = pd.read_csv('dados/test.csv')
treino = pd.read_csv('dados/train.csv')
df_train.head()
df_test.head()
df_train.info()
df_test.info()
treino['Not Survived'] = treino['Survived'].map({0:1, 1:0})
sobreviventes_por_classe = treino.pivot_table(index='Pclass', values=('Survived', 'Not Survived'), aggfunc=np.sum)
sobreviventes_por_classe.head()
sobreviventes_por_classe.plot(kind='bar')
plt.title('sobreviventes por classe')
plt.show()
df_train.drop(['Name','Ticket','Cabin','Fare','Embarked'], axis=1, inplace=True)
df_train.head()
df_train.dropna(subset=['Age'], inplace = True)
df_train.info()
df_test.info()
dummies = pd.get_dummies(df_train.Sex)
df_train.Sex.value_counts()
df_train[['male', 'female']] = dummies
df_train.drop(['Sex'], axis=1, inplace=True)
dummies = pd.get_dummies(df_test.Sex)
df_test.Sex.value_counts()
df_test[['male', 'female']] = dummies
df_train.Pclass.value_counts()
dummies = pd.get_dummies(df_train.Pclass)
df_train.Pclass.value_counts()
df_train[['3', '2', '1']] = dummies
df_train.drop(['Pclass'], axis=1, inplace=True)
dummies = pd.get_dummies(df_test.Pclass)
df_test.Pclass.value_counts()
df_test[['3', '2', '1']] = dummies
df_test.head()
df_train.columns = ['PassengerId','Survived','Age','SibSp','Parch','male', 'female','Lower','Middle','Upper']
df_test.columns = ['PassengerId','Pclass','Name', 'Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','male', 'female','Lower','Middle','Upper']
df_train.SibSp.value_counts()
df_train.Parch.value_counts()
df_train = df_train[df_train.SibSp < 2]
df_train = df_train[df_train.Parch < 3]
df_train.info()
df_train.corr()
features_selecionadas = df_train[['Survived','Age','SibSp','Parch','male','female','Lower','Middle','Upper']].columns
sns.heatmap(df_train[features_selecionadas].corr(), annot=True)
plt.show()
from sklearn.model_selection import train_test_split

features = df_train.columns.difference(['PassengerId','Survived'])

x = df_train[features]
y = df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
parametros = {
    'n_estimators': [4, 8],
    'max_depth': [4, 8],
    'max_features': ['auto', 'log2'],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(rf, param_grid=parametros, cv=5, scoring='recall', n_jobs=-1 )
grid_search.fit(x_train, y_train)
grid_search.best_params_
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy', max_depth=4, max_features='log2', n_estimators=8, n_jobs=-1)
rf.get_params()
from sklearn.metrics import classification_report, confusion_matrix
def gera_resultados_modelo(x_train, x_test, y_train, y_test, modelo):
    print(modelo.__class__.__name__) # nome modelo
    modelo.fit(x_train, y_train) # treinando
    previsao = modelo.predict(x_test) # prevendo
    print(classification_report(y_test, previsao)) ### metrica de performance
    print(confusion_matrix(y_test, previsao))
gera_resultados_modelo(x_train, x_test, y_train, y_test, rf)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
lr = LogisticRegression(class_weight='balanced', random_state=5)
dt = DecisionTreeClassifier(max_depth=5)
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(
    estimators=[('lr',lr),('rf',rf),('dt',dt)], weights=[3,2,1])
gera_resultados_modelo(x_train, x_test, y_train, y_test, vc)
df_test['Age'] = df_test['Age'].transform(lambda x: x.fillna(x.median()))
features_teste = df_test.columns.difference(['PassengerId','Pclass','Name','Sex','Ticket','Fare','Cabin','Embarked'])
#teste = df_test.columns.difference(['PassengerId'])
teste = df_test[features_teste]
pred = vc.predict(teste)
survived = np.array(pred)
gender_submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 
                                  'Survived': survived})
gender_submission.Survived.value_counts()
gender_submission.to_csv('dados/gender_submission.csv', index=False)
arquivo_gerado = pd.read_csv('dados/gender_submission.csv')
arquivo_gerado.head()
arquivo_gerado.tail()
