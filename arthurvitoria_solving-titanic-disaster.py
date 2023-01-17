# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import random

import matplotlib.pyplot as plt



import seaborn as sns



#scikit-learn: bibliotecas com funções para aprendizado de máquina no python

from sklearn.pipeline import Pipeline



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestClassifier



from scipy import stats

from scipy.stats import spearmanr

from scipy.stats import levene

from scipy.stats import shapiro

from scipy.stats import chi2_contingency

from scipy.stats import mannwhitneyu



random.seed(402)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
treino = pd.read_csv("../input/titanic/train.csv") 

teste  = pd.read_csv("../input/titanic/test.csv") 



y_treino = treino['Survived'].copy()

treino.drop('Survived', axis = 1, inplace = True)



print("Inicialmente, no conjunto de treino, temos %d preditores e %d registros." % (treino.shape[1], treino.shape[0]))
treino.info()
treino.head()
treino.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
(treino.isnull().sum()/treino.shape[0]).sort_values(ascending = False)
treino.drop(["Cabin"], axis=1, inplace=True)
treino["Age"].describe()
treino["Embarked"].describe()
imp_median = SimpleImputer(strategy='median')

imp_moda = SimpleImputer(strategy='most_frequent')



col_num = ["Age","Fare","SibSp","Parch"]

col_cat = ["Embarked","Sex","Pclass"]



imp_median.fit(treino[col_num])

treino.Age = imp_median.transform(treino[col_num])



imp_moda.fit(treino[col_cat])

treino.Embarked = imp_moda.transform(treino[col_cat])



treino[col_cat] = treino[col_cat].astype('category')
treino.info()
treino['Sex'].value_counts()
treino['Embarked'].value_counts()
treino['Pclass'].value_counts()
pd.crosstab(treino.Sex, y_treino)
pd.crosstab(treino.Sex, y_treino).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(treino.Embarked, y_treino)
pd.crosstab(treino.Embarked, y_treino).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(treino.Pclass, y_treino)
pd.crosstab(treino.Pclass, y_treino).apply(lambda r: r/r.sum(), axis=1)
tab_cat = pd.DataFrame(index=col_cat, columns=['valor-p','Método'])



for item in col_cat:

    

    aux = np.array(pd.crosstab(treino[item], y_treino))

    

    tab_cat.loc[item,'valor-p'] = chi2_contingency(aux, correction = True)[1]

    tab_cat.loc[item,'Método'] = 'Chi²'



print(tab_cat)
fig, axs = plt.subplots(1,2)

sns.boxplot(x=y_treino, y = treino.Age, ax = axs[0]) # Idade dos passageiros(em anos). 

sns.boxplot(x=y_treino, y = treino.Fare, ax = axs[1]) # Tarifa paga pelo bilhete
fig, axs = plt.subplots(1,2)

sns.boxplot(x=y_treino, y = treino.SibSp, ax = axs[0]) # Número de irmãos / cônjuges a bordo do Titanic

sns.boxplot(x=y_treino, y = treino.Parch, ax = axs[1]) # Número de pais / filhos a bordo do Titanic
treino.Age.groupby(y_treino).describe()
treino.Fare.groupby(y_treino).describe()
treino.SibSp.groupby(y_treino).describe()
treino.Parch.groupby(y_treino).describe()
tab_num = pd.DataFrame(index=col_num, columns=['Shapiro (valor-p)[0]','Shapiro (valor-p)[1]','Levene (valor-p)','Método','valor-p'])



for item in col_num:

    

    alpha = 0.05

    

    lev = levene(treino[item][y_treino == 0], treino[item][y_treino == 1])[1]

    tab_num.loc[item,'Levene (valor-p)'] = lev

    

    shap0 = shapiro(treino[item][y_treino == 0])[1]

    tab_num.loc[item,'Shapiro (valor-p)[0]'] = shap0

    

    shap1 = shapiro(treino[item][y_treino == 1])[1]

    tab_num.loc[item,'Shapiro (valor-p)[1]'] = shap1

    

    if((lev <= alpha) or (shap0 <= alpha) or (shap1 <= alpha)):

        

        tab_num.loc[item,'valor-p'] = mannwhitneyu(treino[item][y_treino == 0], treino[item][y_treino == 1])[1]

        tab_num.loc[item,'Método'] = 'Mann Whitney'

    

    else:

        

        tab_num.loc[item,'valor-p'] = ttest_ind(treino[item][y_treino == 0], treino[item][y_treino == 1])[1]

        tab_num.loc[item,'Método'] = 'Teste-t'



print(tab_num)
cat_sig = tab_cat[tab_cat['valor-p'] <= 0.05].index

cat_sig
num_sig = tab_num[tab_num['valor-p'] <= 0.05].index

num_sig
sig_nor = pd.DataFrame(index = num_sig, columns=['Shapiro (valor-p)'])



for item in num_sig:

    

    alpha = 0.05

    

    sig_nor.loc[item,'Shapiro (valor-p)'] = shapiro(treino[item])[1]

    

print(sig_nor)
treino[num_sig].corr(method ='spearman') 
# Pipeline para tratamento dos dados numéricos e categóricos



features_numericas = num_sig

features_categoricas = cat_sig

features_para_remover = np.setdiff1d(treino.columns,num_sig.append(cat_sig),assume_unique=True).tolist()



numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())])



preprocessor = ColumnTransformer(

    transformers=[

        ('Features numericas', numeric_transformer, features_numericas),

        ('Features categoricas', categorical_transformer, features_categoricas),

        ('Feature para remover', 'drop', features_para_remover)

])
# criar o modelo padrão

rf = RandomForestClassifier(random_state=42)



treino_preprop = preprocessor.fit_transform(treino)
rf.fit(treino_preprop, y_treino)
accuracy_score(y_treino, rf.predict(treino_preprop))
# create the grid

n_estimators = [100,1000, 3000]

max_depth = [None, 5, 10]



param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)



# procurar por grade de parâmetros os valores mais apropriados.

grid = GridSearchCV(estimator=rf, 

                    param_grid=param_grid,

                    cv=3,

                    verbose=2,

                    n_jobs=-1,

                    scoring = 'accuracy')
grid_result = grid.fit(treino_preprop, y_treino)
grid_result.best_params_
arv = Pipeline(steps=[('preprocessor', preprocessor),

                  ('classifier', RandomForestClassifier(**grid_result.best_params_))]) 
arv.fit(treino, y_treino)
accuracy_score(y_treino, arv.predict(treino))
# test our CV score

cross_val_score(rf, treino_preprop, y_treino, cv=5).mean()
cross_val_score(arv, treino, y_treino, cv=5).mean()
y_pred = arv.predict(teste)



submissao = pd.DataFrame({"PassengerId": teste["PassengerId"],

                         "Survived": y_pred

                         })



submissao.to_csv("./submission.csv", index = False)