import pandas as pd

import sklearn

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno

from sklearn.preprocessing import OrdinalEncoder



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing



%matplotlib inline
# importando a base de dados

# 'df' será a nossa base de treino



df = pd.read_csv('../input/adult-pmr3508/train_data.csv',

        engine='python',

        na_values="?")
# número de linhas x colunas do DataFrame

print(df.shape)



# imprime as 5 primeiras linhas do DataFrame

df.head()
# retirando a coluna "Id" do DataFrame



df = df.drop(['Id'], axis=1)
# breve descrição estatística das variáveis numéricas



df.describe()
# histogramas das variáveis numéricas



df.hist(figsize=(20,4), layout=(1,8), color='c')

plt.figure()
# breve descrição das variáveis categóricas e da label



df.describe(exclude = [np.number])
# geração de gráficos e barra das variáveis categóricas e da label



categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race','sex', 'native.country','income']

fig, ax = plt.subplots(3, 3, figsize=(20, 40))

for variable, subplot in zip(categorical, ax.flatten()):

    sns.countplot(df[variable], ax=subplot)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)
# expõe visualmente os dados faltantes



msno.matrix(df)
# informa quantos e quais dados faltantes de cada categoria



df.isnull().sum()
# Preenchendo dados faltantes de variáveis categóricas com suas modas



df["workclass"] = df["workclass"].fillna('Private')

df["occupation"] = df["occupation"].fillna('Prof-specialty')

df["native.country"] = df["native.country"].fillna('United-States')



# informa se há dados faltantes no DataFrame



df.isnull().sum()
# separando o DataFrame da base de treino entre aqueles cuja renda é maior que 50.000 dólares e aqueles cuja renda é menor ou igual



df_menos = df.loc[df['income'] == '<=50K']

df_mais = df.loc[df['income'] == '>50K']



# criando gráficos de barra que relacionam as features com a label

features=['age', 'workclass','fnlwgt', 'education', 'education.num',

          'marital.status', 'occupation', 'relationship', 'race','sex',

          'capital.gain', 'capital.loss','hours.per.week', 'native.country']



for atributo in features:

    uniao = pd.concat([df_menos[atributo].value_counts(), df_mais[atributo].value_counts()], keys=["<=50K", ">50K"], axis=1)

    uniao.plot(kind='bar', xlabel=atributo, ylabel='quantidade', color=['m','c'])

# transformando os dados categóricos em numéricos



features_categoricas = ["sex", "native.country", "workclass", "marital.status", "income", "race", "relationship", "occupation","education"]

df_numericos = df.copy()

enc = OrdinalEncoder()

df_numericos[features_categoricas] = enc.fit_transform(df_numericos[features_categoricas])



#gerando o Heatmap para analisar as correlações entre as variáveis



plt.figure(figsize=(20, 20))

sns.heatmap(df_numericos.corr(), annot=True, cmap="RdYlBu", vmin=-1)
# retirando as colunas que foram consideradas pouco relevantes para o treinamento do modelo

X_train_df = df_numericos.drop(columns = ['workclass','fnlwgt','native.country', 'education','income'])



#criando coluna com o target

Y_train_df = df.income
%%time



classifiers = {}

predictions = {}

scores = {}



scores['KNN'] = 0.0



# aplicando validação cruzada para o melhor descobrir o melhor K



for k in range(25, 35):

    knn = KNeighborsClassifier(k, metric = 'manhattan')

    score = np.mean(cross_val_score(knn, X_train_df, Y_train_df, cv = 10))

    

    if score > scores['KNN']:

        bestK = k

        scores['KNN'] = score

        classifiers['KNN'] = knn



#treinando o modelo 



classifiers['KNN'].fit(X_train_df, Y_train_df)

        

print("Best acc: {}, K = {}".format(scores['KNN'], bestK))
# importando a base de dados de teste



X_test_df= pd.read_csv('../input/adult-pmr3508/test_data.csv',

        engine='python',

        na_values="?")
# substituindo os valores faltantes pela moda



X_test_df["workclass"] = X_test_df["workclass"].fillna('Private')

X_test_df["occupation"] = X_test_df["occupation"].fillna('Prof-specialty')

X_test_df["native.country"] = X_test_df["native.country"].fillna('United-States')
# transformando os dados categóricos em numéricos



test_features_categoricas = ["sex", "native.country", "workclass", "marital.status", "race", "relationship", "occupation","education"]

X_test_df_numericos = X_test_df.copy()

enc = OrdinalEncoder()

X_test_df_numericos[test_features_categoricas] = enc.fit_transform(X_test_df_numericos[test_features_categoricas])
# retirando as colunas que foram consideradas pouco relevantes para o treinamento do modelo



X_test_final = X_test_df_numericos.drop(columns = ['Id','workclass','fnlwgt','native.country', 'education'])
%%time



#treinando a base de teste com KNN-33



predictions['KNN'] = classifiers['KNN'].predict(X_test_final)
# formatando o modelo de submissão da tarefa



submission = pd.DataFrame()

submission[0] = X_test_final.index

submission[1] = predictions['KNN']

submission.columns = ['Id', 'Income']





submission.head()
# criando arquivo .csv da submissão



submission.to_csv("submission.csv", index = False)