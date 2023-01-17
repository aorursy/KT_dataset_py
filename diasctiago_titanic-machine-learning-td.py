import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier,LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import seaborn as sns  
import matplotlib.pyplot as plt  
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head(2)
test.head(2)
sub.head(2)
# Verificar nulos base de treino
train.isna().sum()
# Verificar nulos base de teste
test.isna().sum()
train.shape
# Survived por classe
pd.crosstab(train.Pclass, train.Survived, margins=True).style.background_gradient(cmap='summer_r')
# Survived por classe e sexo
pd.crosstab([train.Sex, train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
# Survived por classe, embarque e sexo 
pd.crosstab([train.Embarked, train.Pclass], [train.Sex, train.Survived], margins=True).style.background_gradient(cmap='summer_r')
# Comparando as faixas de idade
sample = train.loc[(train['Age'] > 62)  & (train['Age'] < 80)]
sns.countplot(x='Age', hue='Survived', data=sample);
# Comparando as faixas de tariva
sample = train.loc[(train['Fare'] > 17) & (train['Fare'] <= 20)]
sns.countplot(x='Fare', hue='Survived', data=sample);
# Criando DF cópia
df = pd.concat((train, test))
# Fazendo limpeza dados nulos treino
# Alterando idades nulas pela mediana
df['Age'].fillna(df['Age'].median(), inplace=True)
# Mediana Fare da class 3
df['Fare'].fillna(df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0], inplace=True)
# Eliminando colunas 
df.drop(columns=['Cabin','Name','Ticket'], inplace=True)
# Descartando 2 observação sem local de embarque
df['Embarked'].dropna(inplace=True)
# Agrupamento de idade criando a coluna Agroup
df['Agroup'] = 0
df.loc[df['Age'] <   1, 'Agroup'] = 1
df.loc[(df['Age'] >= 1)  & (df['Age'] < 2),  'Agroup'] = 2
df.loc[(df['Age'] >= 2)  & (df['Age'] < 3),  'Agroup'] = 3
df.loc[(df['Age'] >= 3)  & (df['Age'] < 7),  'Agroup'] = 4
df.loc[(df['Age'] >= 7)  & (df['Age'] < 12), 'Agroup'] = 5
df.loc[(df['Age'] >= 12) & (df['Age'] < 16), 'Agroup'] = 6
df.loc[(df['Age'] >= 16) & (df['Age'] < 25), 'Agroup'] = 7
df.loc[(df['Age'] >= 25) & (df['Age'] < 35), 'Agroup'] = 8
df.loc[(df['Age'] >= 35) & (df['Age'] < 48), 'Agroup'] = 9
df.loc[(df['Age'] >= 48) & (df['Age'] < 50), 'Agroup'] = 10
df.loc[(df['Age'] >= 50) & (df['Age'] < 62), 'Agroup'] = 11
df.loc[df['Age'] >=  62, 'Agroup'] = 12
# Visualização Gráfica da coluna Agroup
pd.crosstab(df.Agroup, df.Survived, margins=True).style.background_gradient(cmap='summer_r')
# Combinação Pclass & Sex criando a coluna Gclass
df['Gclass']=0
df.loc[((df['Sex'] == 'male')   & (df['Pclass'] == 1)), 'Gclass'] = 1
df.loc[((df['Sex'] == 'male')   & (df['Pclass'] == 2)), 'Gclass'] = 2
df.loc[((df['Sex'] == 'male')   & (df['Pclass'] == 3)), 'Gclass'] = 2
df.loc[((df['Sex'] == 'female') & (df['Pclass'] == 1)), 'Gclass'] = 3
df.loc[((df['Sex'] == 'female') & (df['Pclass'] == 2)), 'Gclass'] = 4
df.loc[((df['Sex'] == 'female') & (df['Pclass'] == 3)), 'Gclass'] = 5
df.loc[(df['Age'] < 1), 'Gclass'] = 6
# Visualização Gráfica da coluna Gclass
pd.crosstab(df.Gclass, df.Survived, margins=True).style.background_gradient(cmap='summer_r')
# Agrupamento da tarifa criando a coluna Fgroup
df['Fgroup'] = 0
df.loc[df['Fare'] <= 0,'Fgroup'] = 0
df.loc[(df['Fare'] > 0) & (df['Fare'] <= 10), 'Fgroup'] = 1
df.loc[(df['Fare'] > 10) & (df['Fare'] <= 20), 'Fgroup'] = 2
df.loc[(df['Fare'] > 20) & (df['Fare'] <= 30), 'Fgroup'] = 3
df.loc[(df['Fare'] > 30) & (df['Fare'] <= 40), 'Fgroup'] = 4
df.loc[(df['Fare'] > 40) & (df['Fare'] <= 50), 'Fgroup'] = 5
df.loc[(df['Fare'] > 50) & (df['Fare'] <= 60), 'Fgroup'] = 6
df.loc[(df['Fare'] > 60) & (df['Fare'] <= 70), 'Fgroup'] = 7
df.loc[(df['Fare'] > 70) & (df['Fare'] <= 90), 'Fgroup'] = 8
df.loc[df['Fare'] > 90, 'Fgroup'] = 9
# Visualização Gráfica da coluna Fgroup
pd.crosstab(df.Fgroup, df.Survived, margins=True).style.background_gradient(cmap='summer_r')
# Agrupamento da tarifa criando a coluna Alone
df['Alone'] = 0
df.loc[(df['SibSp'] > 0) & (df['Sex'] ==   'male'), 'Alone'] = 1
df.loc[(df['Parch'] > 0) & (df['Sex'] ==   'male'), 'Alone'] = 2
df.loc[(df['SibSp'] > 0) & (df['Sex'] == 'female'), 'Alone'] = 3
df.loc[(df['Parch'] > 0) & (df['Sex'] == 'female'), 'Alone'] = 4
df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0) & (df['Sex'] == 'male'), 'Alone'] = 5
df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0) & (df['Sex'] == 'female'), 'Alone'] = 6
# Visualização Gráfica da coluna Alone
pd.crosstab(df.Alone, df.Survived, margins=True).style.background_gradient(cmap='summer_r')
df.head(2)
# Adicionando colunas de prioridade
df['Priority'] = 7 # (7) Others
df.loc[(df['Pclass'] == 2) & (df['Sex'] == 'female'), 'Priority'] = 6 # (6) Women in Pclass 2
df.loc[(df['Pclass'] == 2) & (df['Age'] <= 17), 'Priority'] = 5 # (5) Kids under 17 in Pclass 2 
df.loc[(df['Pclass'] == 1) & (df['Age'] <= 17), 'Priority'] = 4 # (4) Kids under 17 in Pclass 1  
df.loc[(df['Gclass'] == 6), 'Priority'] = 3 # (3) Babies under 1 
df.loc[(df['Gclass'] == 3), 'Priority'] = 2 # (2) Women in Pclass 1 
df.loc[(df['Fgroup'] == 7), 'Priority'] = 1 # (1) Nobles 
# Visualização Gráfica da coluna Priority
pd.crosstab(df.Priority, df.Survived, margins=True).style.background_gradient(cmap='summer_r')
df.info()
columns_base = ['PassengerId','Survived','Pclass','Sex','Alone',
                'Embarked','Gclass','Agroup','Fgroup','Priority']
df_base = df[columns_base].copy()
df_base.dropna(subset=['Embarked'], inplace=True)
df_base.shape
# Correlação das colunas
sns.heatmap(df_base.corr(), annot=True);
# Transformando dados categóricos treino
columns_base.remove('Survived')
columns_base.remove('PassengerId')
df_base = pd.get_dummies(df_base, columns=columns_base)
df_base.shape
# Selecionando features e target
X = df_base.dropna().drop(columns=['Survived','PassengerId'])
y = df_base.dropna().Survived
# Verificando Balanceamento da classificação
y.value_counts()
# Realizar balanceamento com SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Original dataset shape %s' % X.shape[0])
print('Resampled dataset shape %s' % X_res.shape[0])
# Verificar melhor modelo de classificação

# Definindo dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Função para analise dos modelos
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(x_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    matrix = metrics.confusion_matrix(y_test, pred)
    report = metrics.classification_report(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print("*************** Classification Report ***************")
    print(report)
    print("Confusion Matrix")
    sns.heatmap(matrix, annot=True, fmt="d")

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, clf
results = []
print('=' * 80)
print('BaggingClassifier')
results.append(benchmark((BaggingClassifier(RandomForestClassifier()))))

print('=' * 80)
print('KNeighborsClassifier')
results.append(benchmark((KNeighborsClassifier(n_neighbors=10))))
print('=' * 80)
print('LinearSVC')
results.append(benchmark((LinearSVC(max_iter=100000))))
print('=' * 80)
print('RandomForestClassifier')
results.append(benchmark((RandomForestClassifier())))
print('=' * 80)
print("SGDClassifier")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=100)))
print('=' * 80)
print("Perceptron")
results.append(benchmark((Perceptron(max_iter=50))))
print('=' * 80)
print("PassiveAggressiveClassifier")
results.append(benchmark((PassiveAggressiveClassifier(max_iter=50))))
print('=' * 80)
print("RidgeClassifier")
results.append(benchmark((RidgeClassifier())))
print('=' * 80)
print('LogisticRegression')
results.append(benchmark((LogisticRegression(max_iter=1000))))
print('=' * 80)
print("MultinomialNB")
results.append(benchmark(MultinomialNB(alpha=.01)))
print('=' * 80)
print("BernoulliNB")
results.append(benchmark(BernoulliNB(alpha=.01)))
print('=' * 80)
print("ComplementNB")
results.append(benchmark(ComplementNB(alpha=.1)))
print('=' * 80)
print('MLPClassifier')
results.append(benchmark((MLPClassifier(max_iter=1000))))
# Train Ensemble Models
print('=' * 80)
print('ExtraTreesClassifier')
results.append(benchmark((ExtraTreesClassifier(n_estimators=200))))
print('=' * 80)
print('GradientBoostingClassifier')
results.append(benchmark((GradientBoostingClassifier(n_estimators=200))))
print('=' * 80)
print('KNeighborsClassifier')
results.append(benchmark((KNeighborsClassifier())))
print('=' * 80)
print('DecisionTreeClassifier')
results.append(benchmark((DecisionTreeClassifier())))
print('=' * 80)
print('ExtraTreeClassifier')
results.append(benchmark((ExtraTreeClassifier(splitter='best'))))
# Resultado da analise os modelos
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
# Seleção do modelo 
clf = GradientBoostingClassifier(n_estimators=800, loss='exponential', tol=0.0001)
clf.fit(x_train, y_train)
# Identificando melhores hiperparametros

# Valindando performance
predicted = clf.predict(x_test)
print('Performance com os paramentros default\n', metrics.accuracy_score(y_test, predicted))

# Melhores hiperparametros
parameters = {'loss': ('deviance','exponential'),
              'n_estimators': (400,800),
              'tol': (0.0001,0.00001),
              }

gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train, y_train)

print('Melhor score\n', gs_clf.best_score_)
print('Melhores parametros')
pprint(gs_clf.best_params_) 
# Selecionando dados de Teste
df_test = df_base.loc[(df_base.Survived.isnull())].copy()
# Modelo definido treinamento com dados completo
clf.fit(X_res, y_res)
# Predição nos dados de teste
predicted_test = clf.predict(df_test.drop(columns=['Survived','PassengerId']))
df_test['Survived'] = pd.to_numeric(predicted_test, downcast='integer')
submission = df_test[['PassengerId','Survived']]
submission.to_csv('submission.csv', index=False)
submission.head()