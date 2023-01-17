%matplotlib inline

import pandas as pd

import numpy as np

import pylab as plt



# define tamanho padrão das figuras

plt.rc('figure', figsize=(10, 5))

fizsize_with_subplots = (10, 10)

bin_size = 10 
# Data Import

df_train = pd.read_csv('/kaggle/input/train.csv')

df_test = pd.read_csv('/kaggle/input/test.csv')





#Databases for Treatment

t_df_train = df_train

t_df_test = df_test



#Displaying Data Info

df_train.head(5)
df_train.info()
df_test.head(5)
df_test.info()
#Train Data

fig = plt.figure(figsize=fizsize_with_subplots) 

fig_dims = (3, 4)



# Survived

plt.subplot2grid(fig_dims, (0, 0))

df_train['Survived'].value_counts().plot(kind='bar', title='Survived')



# Pclass

plt.subplot2grid(fig_dims, (0, 1))

df_train['Pclass'].value_counts().plot(kind='bar', title='Pclass')





# Sex

plt.subplot2grid(fig_dims, (0, 2))

df_train['Sex'].value_counts().plot(kind='bar', title='Sex')





# Age

plt.subplot2grid(fig_dims, (0, 3))

df_train['Age'].hist()

plt.title('Age')



# SibSp

plt.subplot2grid(fig_dims, (1, 0))

df_train['SibSp'].value_counts().plot(kind='bar', title='SibSp')





# Parch

plt.subplot2grid(fig_dims, (1, 1))

df_train['Parch'].value_counts().plot(kind='bar', title='Parch')





# Ticket

plt.subplot2grid(fig_dims, (1, 2))

df_train['Ticket'].value_counts().plot(kind='bar', title='Ticket')





# Fare

plt.subplot2grid(fig_dims, (1, 3))

df_train['Fare'].value_counts().plot(kind='bar', title='Fare')





# Cabin

plt.subplot2grid(fig_dims, (2, 0))

df_train['Cabin'].value_counts().plot(kind='bar', title='Cabin')





# Embarked

plt.subplot2grid(fig_dims, (2, 1))

df_train['Embarked'].value_counts().plot(kind='bar', title='Embarked')

#Test Data

fig = plt.figure(figsize=fizsize_with_subplots) 

fig_dims = (3, 4)



# Pclass

plt.subplot2grid(fig_dims, (0, 1))

df_test['Pclass'].value_counts().plot(kind='bar', title='Pclass')





# Sex

plt.subplot2grid(fig_dims, (0, 2))

df_test['Sex'].value_counts().plot(kind='bar', title='Sex')





# Age

plt.subplot2grid(fig_dims, (0, 3))

df_test['Age'].hist()

plt.title('Age')



# SibSp

plt.subplot2grid(fig_dims, (1, 0))

df_test['SibSp'].value_counts().plot(kind='bar', title='SibSp')





# Parch

plt.subplot2grid(fig_dims, (1, 1))

df_test['Parch'].value_counts().plot(kind='bar', title='Parch')





# Ticket

plt.subplot2grid(fig_dims, (1, 2))

df_test['Ticket'].value_counts().plot(kind='bar', title='Ticket')





# Fare

plt.subplot2grid(fig_dims, (1, 3))

df_test['Fare'].value_counts().plot(kind='bar', title='Fare')





# Cabin

plt.subplot2grid(fig_dims, (2, 0))

df_test['Cabin'].value_counts().plot(kind='bar', title='Cabin')





# Embarked

plt.subplot2grid(fig_dims, (2, 1))

df_test['Embarked'].value_counts().plot(kind='bar', title='Embarked')
#Train - Age:

for pc in sorted(t_df_train['Pclass'].unique()):

    for sex in t_df_train['Sex'].unique():

        loc = (t_df_train['Pclass'] == pc) & (t_df_train['Sex'] == sex)

        media = t_df_train[loc]['Age'].mean()

        nulos = t_df_train[loc]['Age'].isnull().sum()

        conhecidos = len(t_df_train[loc]['Age']) - nulos

        print('Classe {}, gênero {} --> {:.1f}'.format(pc, sex, media))

        print('Idade conhecida: {}, desconhecida: {}'.format(conhecidos, nulos))

        t_df_train.loc[loc & t_df_train[loc]['Age'].isnull(), 'Age'] = media  # preenchimento

        

t_df_train.info()
#Train - Cabin:

for pc in sorted(t_df_train['Pclass'].unique()):

    loc = (t_df_train['Pclass'] == pc) & (t_df_train['Cabin'].isnull())

    t_df_train.loc[loc, 'Cabin'] = 'Unk ' + str(pc)

        

t_df_train.info()

t_df_train.head(20)
#Train - Embarked:

t_df_train['Embarked'].fillna('S', inplace=True)

t_df_train.info()
#Test - Age

for pc in sorted(t_df_test['Pclass'].unique()):

    for sex in t_df_test['Sex'].unique():

        loc = (t_df_test['Pclass'] == pc) & (t_df_test['Sex'] == sex)

        media = t_df_test[loc]['Age'].mean()

        nulos = t_df_test[loc]['Age'].isnull().sum()

        conhecidos = len(t_df_test[loc]['Age']) - nulos

        print('Classe {}, gênero {} --> {:.1f}'.format(pc, sex, media))

        print('Idade conhecida: {}, desconhecida: {}'.format(conhecidos, nulos))

        t_df_test.loc[loc & t_df_test[loc]['Age'].isnull(), 'Age'] = media  # preenchimento

        

t_df_train.info()
#Test - Fare:

for pc in sorted(t_df_test['Pclass'].unique()):

    #for sex in t_df_test['Sex'].unique():

        loc = (t_df_test['Pclass'] == pc) #& (t_df_test['Sex'] == sex)

        media = t_df_test[loc]['Fare'].mean()

        nulos = t_df_test[loc]['Fare'].isnull().sum()

        conhecidos = len(t_df_test[loc]['Fare']) - nulos

        #print('Classe {}, gênero {} --> {:.1f}'.format(pc, sex, media))

        print('Fare conhecida: {}, desconhecida: {}'.format(conhecidos, nulos))

        t_df_test.loc[loc & t_df_test[loc]['Fare'].isnull(), 'Fare'] = media  # preenchimento

        

t_df_train.info()
#Test - Cabin:

for pc in sorted(t_df_test['Pclass'].unique()):

    loc = (t_df_test['Pclass'] == pc) & (t_df_test['Cabin'].isnull())

    t_df_test.loc[loc, 'Cabin'] = 'Unk ' + str(pc)

        

t_df_test.info()

t_df_test.head(20)
t_df_train.info()

t_df_test.info()

t_df_train = t_df_train.drop(['PassengerId', 'Name'], axis=1)

t_df_test = t_df_test.drop(['PassengerId', 'Name'], axis=1)
#Dummy for training data

novas_colunas_ticket = pd.get_dummies(t_df_train['Ticket'], prefix='ticket') 

novas_colunas_pclass = pd.get_dummies(t_df_train['Pclass'], prefix='class') 

novas_colunas_sex = pd.get_dummies(t_df_train['Sex'], prefix='sex') 

novas_colunas_embarked = pd.get_dummies(t_df_train['Embarked'], prefix='embarked') 

novas_colunas_embarked = pd.get_dummies(t_df_train['Cabin'], prefix='cabin') 



t_df_train = pd.concat([t_df_train, novas_colunas_pclass, novas_colunas_sex, novas_colunas_embarked, novas_colunas_embarked, novas_colunas_ticket], axis=1)

t_df_train.drop(['Pclass', 'Sex', 'Embarked', 'Cabin', 'Ticket'], axis=1, inplace=True)



#Dummy for test data

novas_colunas_ticket = pd.get_dummies(t_df_test['Ticket'], prefix='ticket') 

novas_colunas_pclass = pd.get_dummies(t_df_test['Pclass'], prefix='class') 

novas_colunas_sex = pd.get_dummies(t_df_test['Sex'], prefix='sex') 

novas_colunas_embarked = pd.get_dummies(t_df_test['Embarked'], prefix='embarked') 

novas_colunas_embarked = pd.get_dummies(t_df_test['Cabin'], prefix='cabin') 



t_df_test = pd.concat([t_df_test, novas_colunas_pclass, novas_colunas_sex, novas_colunas_embarked, novas_colunas_embarked, novas_colunas_ticket], axis=1)

t_df_test.drop(['Pclass', 'Sex', 'Embarked', 'Cabin', 'Ticket'], axis=1, inplace=True)



t_df_train.head(5)

t_df_test.head(5)
y_train = t_df_train['Survived'].values

t_df_train.drop('Survived', axis=1, inplace=True)



X_train, X_test = t_df_train.values, t_df_test.values
from sklearn.model_selection import KFold

np.random.seed(5)

kf = KFold(n_splits=5, shuffle=True, random_state=5)





#Função idêntica à usada nos modelos de regressão.

def avalia_classificador(clf, kf, X, y, f_metrica):

    metrica_val = []

    metrica_train = []

    for train, valid in kf.split(X,y):

        x_train = X[train]

        y_train = y[train]

        x_valid = X[valid]

        y_valid = y[valid]

        clf.fit(x_train, y_train)

        y_pred_val = clf.predict(x_valid)

        y_pred_train = clf.predict(x_train)

        metrica_val.append(f_metrica(y_valid, y_pred_val))

        metrica_train.append(f_metrica(y_train, y_pred_train))

    return np.array(metrica_val).mean(), np.array(metrica_train).mean()





def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):

    c = 100.0 if percentual else 1.0

    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))

    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))
from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score



lr = LogisticRegression(solver='liblinear')

media_acuracia_val, media_acuracia_train = avalia_classificador(lr, kf, X_train, y_train, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(lr, kf, X_train, y_train, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)



dt = tree.DecisionTreeClassifier(max_depth=3)

media_acuracia_val, media_acuracia_train = avalia_classificador(dt, kf, X_train, y_train, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

media_auc_val, media_auc_train = avalia_classificador(dt, kf, X_train, y_train, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
preds = dt.fit(X_train, y_train).predict(X_train)



resultado = {'PassengerId': df_train['PassengerId'], 'Survived': preds.astype('int')}

resultado = pd.DataFrame.from_dict(resultado)
resultado.info()

resultado.head()
resultado.to_csv('resultado.csv', index=False, header=True)