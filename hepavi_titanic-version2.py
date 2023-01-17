# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = os.path.join(dirname,'train.csv')
df = pd.read_csv(path, delimiter = ",", index_col = 'PassengerId')
# Construção de reporte com a visão geral dos dados
from pandas_profiling import ProfileReport

# profile = ProfileReport(df, minimal=True) # Caso os dados forem grandes
profile = ProfileReport(df, title='Titanic',html={'style':{'full_width':True}})
profile.to_notebook_iframe()
# Complementação da visualização dos dados por meio de alguns gráficos
import seaborn as sns
import matplotlib.pyplot as plt

col = ['Survived', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']
sns.set(style="ticks", color_codes=True)
sns.pairplot(df[col], hue = "Survived", corner=True)
data  = 'Survived'
data1 = 'Sex'
data2 = 'Embarked'
data3 = 'Parch'
data4 = 'SibSp'

sns.set(style="darkgrid", palette = 'pastel')
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
sns.countplot(df[data1], hue=df[data], ax=axs[0,0])
sns.countplot(df[data2], hue=df[data], ax=axs[0,1])
sns.countplot(df[data3], hue=df[data], ax=axs[1,0])
sns.countplot(df[data4], hue=df[data], ax=axs[1,1])

plt.subplots_adjust(wspace=0.2, hspace=0.3)
data1 = df.Age[df.Survived == 0]
data2 = df.Age[df.Survived == 1]

num_bins = 25

freq, bins, patches = plt.hist([data1,data2], num_bins, alpha=0.7)
plt.title("Survived for Age") 
plt.show()
print("Bins:", bins)
data1 = df.Fare[df.Survived == 0]
data2 = df.Fare[df.Survived == 1]

bins = np.linspace(0, 300, num=20)

freq, bins, patches = plt.hist([data1,data2], bins, alpha=0.7)
plt.title("Survived for Fare") 
plt.show()

print("Bins:", bins)
np.sum(freq,axis=1)
# Antes de começar o tratamento dos dados vou incluir as informações de teste para que as
# mesmas alterações fiquem nos dois conjuntos de dados.

path = os.path.join(dirname,'test.csv')
test_df = pd.read_csv(path, delimiter = ",", index_col = 'PassengerId')
test_df['type'] = "test"

df['type'] = "train"

df1 = df.append(test_df)
print(df1.shape)
# A variável nome é composto pelo título mais o nome completo do passageiro, 
# vamos segrega-lo para observar melhor essas informações.

# Cria as variáveis Lastname (sobrenome), Title (Mrs, miss, ...) e Firstname (restante do nome)
df1[["Lastname","AUX"]] = df1.Name.str.split(",", n=1,expand = True)
df1[["Title","Firstname"]] = df1.AUX.str.split(".", n=1,expand = True)
df1 = df1.drop(["AUX"], axis=1)
# Verificar a frequencia de passageiros por título
df1.Title.value_counts()
# Agrupa titulo entre Mr, Miss, Mrs, Mastes e Outros
aux = df1.Title.value_counts()
aux1 = list(aux[aux.values>10].index)
df1.Title = df1.Title.where(df1["Title"].isin(aux1),other = "Outros")
df1.Title.value_counts()
# Observando as variáveis Parch e SibSp notamos que as piores porcentagem de morte ocorre
# quando seus valores são zero. Nos induzindo a concluir que os passageiros sem familia 
# tem chance maior de não sobreviver.

# Cria a variável Family. Se 1 possuia mais entes embarcado e 0 CC.
df1['N_members'] = df1['SibSp'] + df1['Parch']
df1['Family'] = df1['N_members'].map(lambda s: 0 if s == 0 else 1)
df1.isnull().sum()
# Embarked
# Como são apenas 2 valores vou preencher com a informação mais frequente S
df1.Embarked.fillna("S", inplace = True)
# Fare
# Percebam que com a inclusão dos dados de teste surgiu um valor missing na variável Fare
# Devida a uma certa correlação com Pclass vou preencher esse valor com a mediana considerando a classe
c = df1.Pclass[df1.Fare.isnull()] 
gf = df1.Fare[df1.Pclass==c.values[0]].mode()
df1.Fare.fillna(gf[0], inplace = True)
# Age
# Dada a grande quantidade de dados ausentes vou utilizar uma regressão linear para 
# preencher os dados faltantes.
# Para isso utilizarei como parâmetros Pclass, SibSp, Parch e Title

col_age = ['Age', 'Pclass', 'SibSp', 'Parch', 'Title']
df2 = df1[col_age]

s = (df2.dtypes == "object")
object_cols = s[s].index

# Trasformação do dado categorico, Title, em uma matriz binária
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)
df_OHE = OHE.fit_transform(df2[object_cols])
df_OHE = pd.DataFrame(data = df_OHE)
df_OHE.index = df2.index

df2_num = df2.drop(object_cols, axis = 1)
df2 = pd.concat([df2_num,df_OHE], axis = 1)

# Separação dos dados de treinamento e teste
X_age_train = df2[~df2.Age.isnull()]
Y_age_train = X_age_train["Age"]
X_age_train = X_age_train.drop(["Age"],axis = 1 )

X_age_test  = df2[df2.Age.isnull()]
Y_age_test  = X_age_test["Age"]
X_age_test  = X_age_test.drop(["Age"],axis = 1 )

# Aplica a regressão linear nos dados de treino
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression()
reg.fit(X_age_train, Y_age_train)
pred = reg.predict(X_age_train)
print('R2:', reg.score(X_age_train, Y_age_train))
print('MSE:', mean_squared_error(Y_age_train,pred))

# Aplica a regressão aos dados que queremos prever
pred_age = reg.predict(X_age_test)

age_df = pd.DataFrame(data=pred_age, columns = ["n_Age"])
age_df.index = X_age_test.index

df1 = df1.merge(age_df, how = 'left', left_index = True, right_index=True, suffixes=(False, False))
df1.Age = df1.Age.fillna(df1.n_Age)
df1 = df1.drop(["n_Age"], axis = 1)
#sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3")

col1 = ['SibSp', 'Parch', 'N_members', 'Family']
col2 = ['Age', 'Fare']

fig, axes = plt.subplots(nrows = 1,ncols = 2,figsize = (12,4), dpi=85)
sns.boxplot(data = df1[col1], width=0.5, palette="Set3", ax = axes[0])
sns.boxplot(data = df1[col2], width=0.5, palette="Set3", ax = axes[1])
# Preparação dos dados separando em treino e teste
surv_col = ['Survived', 'Pclass', 'Sex', 'Embarked', 'type', 'Age', 'Title', 
            'SibSp', 'Parch', 'Fare', 'N_members', 'Family']

object_cols = ['Sex', 'Embarked', 'Title']

df2 = df1[surv_col]

OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)
df_OHE = OHE.fit_transform(df2[object_cols])
df_OHE = pd.DataFrame(data = df_OHE)
df_OHE.index = df2.index

df2_num = df2.drop(object_cols, axis = 1)
df2 = pd.concat([df2_num,df_OHE], axis = 1)

X_train = df2[df2.type == 'train']
Y_train = X_train['Survived']
X_train = X_train.drop(['Survived', 'type'],axis = 1)

X_test = df2[df2.type == 'test']
X_test = X_test.drop(['Survived', 'type'],axis = 1)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators = 50, learning_rate=0.1, min_samples_leaf=8, subsample=0.8)
clf.fit(X_train, Y_train)

feature_importances = pd.DataFrame(clf.feature_importances_,index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances
# Uso do GridSearchCV para a seleção dos melhores parâmetros no conjunto de treino
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [5, 10, 50],
              'min_samples_split': [5, 10, 15, 20],
              'max_depth': [30, 35, 40, 50],
              'max_features': ['auto',0.5, 1]
             }
clf = GradientBoostingClassifier(learning_rate=0.1, subsample=0.8)
GS = GridSearchCV(estimator = clf, param_grid = param_grid, cv=6)
GS.fit(X_train, Y_train)
print('Best estimator:', GS.best_estimator_)
print('Best param:', GS.best_params_)
print('Score:', GS.best_score_)
# Aplicação do modelo no conjunto de teste
clf = GradientBoostingClassifier(learning_rate=0.1, 
                                 subsample=0.8, 
                                 n_estimators = 10,
                                 min_samples_split =  20,
                                 max_depth = 40,
                                 max_features = 0.5)
clf.fit(X_train, Y_train)
pred = pd.DataFrame(clf.predict(X_test), columns = ['Survived'], index = X_test.index).astype('Int64')
pred