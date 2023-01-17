import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import itertools

import math

import category_encoders as ce
# Leitura dos dados

X_train = pd.read_csv('../input/titanic/train.csv')



X_test = pd.read_csv('../input/titanic/test.csv')



#Visualização

X_train.head(10)
print("Formato dos dados de treino: {}".format(X_train.shape))

print("Formato dos dados de teste: {}".format(X_test.shape))



print("Features iniciais contidas no dataset: {}".format(X_test.columns))

print("Label: Survived")
X_train.describe()
# Criando a representação, área de plot

fig1, ax1 = plt.subplots(figsize = (4,4))



# Conjunto de dados a ser representado

sns.set(style="darkgrid")

sizes = X_train.Survived.value_counts()

labels = ['Não sobreviveu', 'Sobreviveu']



# Criando o gŕafico

ax1.pie(sizes, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90, colors = [(0.8,0.4,0.4), 'skyblue'])



# Opções Adicionais

plt.title('Taxa de sobrevivência no Titanic')

ax1.axis('equal')



# Mostrando o gŕafico

plt.show()
sizes = X_train.Sex.value_counts()

labels = ['Homens', 'Mulheres']



# Criando a representação, área de plot

fig, (ax2,ax1) = plt.subplots(figsize = [12,12], nrows = 1, ncols = 2)

plt.subplots_adjust(left=0, bottom=None, right=1, top=0.5, wspace = 0.2, hspace=None)

# Gráfico de pizza (porcentagem de homens e de mulheres)

sns.set(style="darkgrid")





ax2.bar(labels,sizes,0.2,0.2, color = ['skyblue', 'pink'])

ax2.set_title('Quantidade Absoluta de Homens e de Mulheres', fontsize = 10)

ax2.set_ylabel('Quantidade',fontsize = 10)





sns.set(style="darkgrid")

sns.countplot(x = 'Survived', hue = 'Sex', data = X_train, ax = ax1)

ax1.set_title('Relação do Sexo do Passageiro com sua Sobrevivência')

ax1.set_xlabel('Sobreviveu', fontsize = 10)

ax1.set_ylabel('Quantidade', fontsize = 10)





# Mostrando o gŕafico

plt.show()
def age_groups(row):

    if row.Age < 20:

        return 1

    elif row.Age < 30:

        return 2

    elif row.Age < 40:

        return 3

    elif row.Age < 60:

        return 4

    elif row.Age < 100:

        return 5

    

X_train['Age_group'] = X_train.apply(lambda row: age_groups(row), axis = 1)

# Criando a representação, área de plot

fig, (ax2,ax1) = plt.subplots(figsize = [12,12], nrows = 1, ncols = 2)

plt.subplots_adjust(left=0, bottom=None, right=1, top=0.5, wspace = 0.2, hspace=None)

# Gráfico de pizza (porcentagem de homens e de mulheres)

sns.set(style="darkgrid")



# Distribuição de Idade dos Passageiros

sns.kdeplot(data=X_train['Age'], shade=True, ax = ax2)

ax2.set_title('Distribuição de Idade dos Passageiros', fontsize = 10)

ax2.set_ylabel('Função distribuição',fontsize = 10)

ax2.set_xlabel('Idade',fontsize = 10)





# Relação da Faixa Etária com a Sobrevivência

sns.countplot(x = 'Survived', hue = 'Age_group', data = X_train, ax = ax1)



ax1.set_title('Relação da Faixa Etária do Passageiro com sua Sobrevivência')

ax1.set_xlabel('Sobreviveu', fontsize = 10)

ax1.set_ylabel('Quantidade', fontsize = 10)







# Mostrando o gŕafico

plt.show()
sizes = X_train.Pclass.value_counts().reindex(index = [1,2,3])

labels = ['1','2','3']



# Criando a representação, área de plot

fig, (ax2,ax1) = plt.subplots(figsize = [12,12], nrows = 1, ncols = 2)

plt.subplots_adjust(left=0, bottom=None, right=1, top=0.5, wspace = 0.2, hspace=None)

# Gráfico de pizza (porcentagem de homens e de mulheres)

sns.set(style="darkgrid")





ax2.bar(labels,sizes,0.2,0.2, color = ['skyblue', 'pink','orange'])

ax2.set_title('Quantidade de Passageiros por Classe de Passagem', fontsize = 10)

ax2.set_ylabel('Quantidade',fontsize = 10)

ax2.set_xlabel('Classe de Passagem',fontsize = 10)







sns.set(style="darkgrid")

sns.countplot(x = 'Survived', hue = 'Pclass', data = X_train, ax = ax1)

plt.title('Relação da Classe da Passagem do Passageiro e sua Sobrevivência')

ax1.set_xlabel('Sobreviveu', fontsize = 10)

ax1.set_ylabel('Quantidade',fontsize = 10)





# Mostrando o gŕafico

plt.show()
correlation = X_train.corr()



plt.figure(figsize=(8,8))

sns.heatmap(correlation, annot = True)

plt.title('Correlação das Features')



plt.show()
missing_val_count_by_column = (X_train.isnull().sum())

print('Features com dados faltantes no DataSet:')

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Substituição dos valores faltantes na coluna de idade pela mediana dos valores

X_train.loc[X_train.Age.isnull(),'Age']=X_train.Age.median()



# Substituição dos valores faltantes na coluna Embarked pelo valor mais comum

X_train.loc[X_train.Embarked.isnull(), 'Embarked'] = X_train.Embarked.describe().top



# Atualização da coluna Age_group

X_train['Age_group'] = X_train.apply(lambda row: age_groups(row), axis = 1)



# Retiramos a variável Cabin, já que possui muitos valores faltantes

X_train.drop(['Cabin'], axis = 1, inplace = True)



# Realizamos uma cópia do Dataset para podermos construir um tratamento e um modelo base

X_train_baseline = X_train.copy()
X_train_baseline.head()
s = (X_train_baseline.dtypes == 'object')

object_cols = list(s[s].index)



print("Variáveis categóricas presentes no conjunto de dados: {}".format(object_cols))
X_train_baseline.describe(include=[np.object])
# Retirada das variáveis Name e Ticket

X_train_baseline.drop(['Name', 'Ticket'], axis = 1, inplace = True)

# Colunas a sofrerem o OH Encoding

object_cols = ['Sex', 'Embarked']



# Aplicação do OH Encoding

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_baseline[object_cols]))



# Recolocando o index

OH_cols_train.index = X_train_baseline.index

X_train_baseline.drop(['Sex', 'Embarked'], axis = 1, inplace=  True)



# Concatenando de volta o dataset

X_train_baseline = pd.concat([X_train_baseline,OH_cols_train ], axis = 1)
def fit_predict_evaluate_model(clf, X, y):

    # Predicting the data

    my_pipeline = clf

    accuracy = 1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='accuracy')

    precision = 1 * cross_val_score(my_pipeline, X, y,

                          cv=5,

                          scoring='precision')

    recall = 1 * cross_val_score(my_pipeline, X, y,

                          cv=5,

                          scoring='recall')

    

    auc = 1 * cross_val_score(my_pipeline, X, y,

                          cv=5,

                          scoring='roc_auc')

    return accuracy.mean(), precision.mean(), recall.mean(), auc.mean()
X = X_train_baseline.drop(['Survived', 'PassengerId'], axis = 1)

y = X_train_baseline['Survived']
# Definição do modelo

randomforest = RandomForestClassifier(random_state = 5, criterion = 'gini', max_depth = 10, max_features = 'auto', n_estimators = 500)



# Métricas de avaliação

acc_RF, precision_RF, recall_RF, auc_RF = fit_predict_evaluate_model(randomforest,X,y)



print("Acurácia:\t {}".format(acc_RF))

print("Precisão:\t {}".format(precision_RF))

print("Recall:\t\t {}".format(recall_RF))



print("AUC:\t\t {}".format(auc_RF))
# Definição do modelo

gbk = GradientBoostingClassifier()

acc_gbk, precision_gbk, recall_gbk, auc_gbk = fit_predict_evaluate_model(gbk,X,y)



# Métricas de avaliação

print("Acurácia:\t {}".format(acc_gbk))

print("Precisão:\t {}".format(precision_gbk))

print("Recall:\t\t {}".format(recall_gbk))



print("AUC:\t\t {}".format(auc_gbk))

# Definição do modelo

lightbm = lgb.LGBMClassifier(num_leaves = 64, objective= 'binary', seed = 7)





# Métricas de avaliação

acc_lgb, precision_lgb, recall_lgb, auc_lgb = fit_predict_evaluate_model(lightbm,X,y)



print("Acurácia:\t {}".format(acc_lgb))

print("Precisão:\t {}".format(precision_lgb))

print("Recall:\t\t {}".format(recall_lgb))

print("AUC:\t\t {}".format(auc_lgb))

# Criando DataFrame



modelos = ['RandomForests', 'XGBoost', 'LGBM']

acuracia = [acc_RF, acc_gbk, acc_lgb]

precision = [precision_RF, precision_gbk, precision_lgb]

recall = [recall_RF, recall_gbk, recall_lgb]

AUC = [auc_RF, auc_gbk, auc_lgb]

initial_results = pd.DataFrame({'Modelo': modelos, 'Acurácia': acuracia, 'Precisão': precision, 'Recall': recall,

                               'AUC': AUC})



initial_results
features = ['Sex', 'Age', 'Pclass']



# Criando interações entre features

interactions = pd.DataFrame(index = X_train.index)



for col1, col2 in itertools.combinations(features,2):

    new_col_name = '_'.join([col1,col2])

    

    new_values = X_train[col1].map(str) + "_" + X_train[col2].map(str)

    interactions[new_col_name] = new_values



# Adicionando interações ao dataframe

X_optimization = X_train.join(interactions)



X_optimization.head()
# Colunas a sofrerem o encoding

object_cols = ['Sex', 'Embarked', 'Sex_Age', 'Sex_Pclass', 'Age_Pclass']



# Colunas retiradas do dataset

X_optimization.drop(['Name', 'Ticket'], axis = 1, inplace = True)







# Criando o count encoder

count_enc = ce.CountEncoder(cols=object_cols)



# Aprendendo e aplicando o encoding

count_enc.fit(X_optimization[object_cols])

X_encoded = count_enc.transform(X_optimization[object_cols])

X_optimization = X_optimization.join(X_encoded.add_suffix('_count'))

X_optimization.drop(object_cols, axis = 1, inplace = True)

X_optm = X_optimization.drop(['Survived', 'PassengerId'], axis = 1)

y_optm = X_optimization['Survived']
randomforest = RandomForestClassifier(random_state = 5, criterion = 'gini', max_depth = 10, max_features = 'auto', n_estimators = 500)

randomforest = randomforest.fit(X, y)



acc_RF_opt, precision_RF_opt, recall_RF_opt, auc_RF_opt = fit_predict_evaluate_model(randomforest,X_optm,y_optm)



print("Acurácia:\t {}".format(acc_RF_opt))

print("Precisão:\t {}".format(precision_RF_opt))

print("Recall:\t\t {}".format(recall_RF_opt))



print("AUC:\t\t {}".format(auc_RF_opt))
gbk_opt = GradientBoostingClassifier()

acc_gbk_opt, precision_gbk_opt, recall_gbk_opt, auc_gbk_opt = fit_predict_evaluate_model(gbk_opt,X_optm,y_optm)



print("Acurácia:\t {}".format(acc_gbk_opt))

print("Precisão:\t {}".format(precision_gbk_opt))

print("Recall:\t\t {}".format(recall_gbk_opt))



print("AUC:\t\t {}".format(auc_gbk_opt))

print("AUC:\t\t {}".format(auc_gbk))
lightbm = lgb.LGBMClassifier(num_leaves = 64, objective= 'binary', seed = 7)







acc_lgb_opt, precision_lgb_opt, recall_lgb_opt, auc_lgb_opt = fit_predict_evaluate_model(lightbm,X_optm,y_optm)



print("Acurácia:\t {}".format(acc_lgb_opt))

print("Precisão:\t {}".format(precision_lgb_opt))

print("Recall:\t\t {}".format(recall_lgb_opt))

print("AUC:\t\t {}".format(auc_lgb_opt))

modelos_opt = ['RandomForests_opt', 'XGBoost_opt', 'LGBM_opt']

acuracia_opt = [acc_RF_opt, acc_gbk_opt, acc_lgb_opt]

precision_opt = [precision_RF_opt, precision_gbk_opt, precision_lgb_opt]

recall_opt = [recall_RF_opt, recall_gbk_opt, recall_lgb_opt]

AUC_opt = [auc_RF_opt, auc_gbk_opt, auc_lgb_opt]

final_results = pd.DataFrame({'Modelo': modelos_opt, 'Acurácia': acuracia_opt, 'Precisão': precision_opt, 'Recall': recall_opt,

                               'AUC': AUC_opt})





pd.concat([initial_results, final_results])