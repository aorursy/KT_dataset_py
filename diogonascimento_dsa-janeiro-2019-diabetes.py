#import

from imblearn.over_sampling import SMOTE

from pandas import read_csv

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")
#dataset treino

df=read_csv('../input/dataset_treino.csv')

df.head()
df.describe()
#distribuição das classes

df.groupby('classe').size()
import seaborn as sns

# Pairplot

sns.pairplot(df)
#subistitui os valores 0 pela media, exceto a variavel num_gestacoes

median_bmi = df['bmi'].median()

df['bmi'] = df['bmi'].replace(to_replace=0, value=median_bmi)

median_pressao_sanguinea = df['pressao_sanguinea'].median()

df['pressao_sanguinea'] = df['pressao_sanguinea'].replace(to_replace=0, value=median_pressao_sanguinea)

median_glicose = df['glicose'].median()

df['glicose'] = df['glicose'].replace(to_replace=0, value=median_glicose)

median_grossura_pele = df['grossura_pele'].median()

df['grossura_pele'] = df['grossura_pele'].replace(to_replace=0, value=median_grossura_pele)

median_insulina = df['insulina'].median()

df['insulina'] = df['insulina'].replace(to_replace=0, value=median_insulina)
balanced_df=df
# Correlação de Pearson

balanced_df.corr(method = 'pearson').sort_values('classe',ascending=False)
#ordena pelo id só pra visualizar como ficou apos a subistituiçao de 0 por media

balanced_df=balanced_df.sort_values('id',ascending=True)

balanced_df.head()
#scala e testando variaveis preditoras

from sklearn.preprocessing import StandardScaler

colunas=['num_gestacoes','glicose','pressao_sanguinea','grossura_pele','insulina','bmi','indice_historico','idade']

#colunas=['glicose','bmi','idade','num_gestacoes','grossura_pele']

#colunas=['glicose','bmi','idade','num_gestacoes']

balanced_padronized_df=balanced_df[colunas]
#confere as variaveis

balanced_padronized_df.head()
# definindo X e Y

X = balanced_padronized_df

Y = balanced_df.classe

# Gerando a nova scala

scaler = StandardScaler().fit(X)

X = scaler.transform(X)

X
#balanceando a classe, classe minotaria 1  208 , classe majoritoria 0    392



from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#balanceamento de classe com smote

#https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

sm = SMOTE(random_state=1)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_train_res
print("Apos balanceamento, classe '1': {}".format(sum(y_train_res==1)))

print("Apos balanceamento, classe '0': {}".format(sum(y_train_res==0)))
#import Keras

#rede neural

#mehlores resultados com adam e binary_crossentropy

from keras.models import Sequential

from keras.layers import Activation, Dense

model = Sequential()

model.add(Dense(1024, activation='relu', input_dim=8, init='uniform'))

model.add(Dense(512, activation='relu',init='uniform'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_res,y_train_res, epochs=92, batch_size=10000, validation_data=(X_train_res, y_train_res))
#data set teste

df_teste=read_csv('../input/dataset_teste.csv')

df_teste.head()
#criando df de treino

from sklearn.preprocessing import StandardScaler

colunas=['num_gestacoes','glicose','pressao_sanguinea','grossura_pele','insulina','bmi','indice_historico','idade']

df_teste_balanced_padronized_df=df_teste[colunas]

df_teste_balanced_padronized_df.describe()
#subistituindo 0 pela media exceto o numero de gestações

df_teste_balanced_padronized_df['bmi'] = df_teste_balanced_padronized_df['bmi'].replace(to_replace=0, value=median_bmi)

df_teste_balanced_padronized_df['pressao_sanguinea'] = df_teste_balanced_padronized_df['pressao_sanguinea'].replace(to_replace=0, value=median_pressao_sanguinea)

df_teste_balanced_padronized_df['glicose'] = df_teste_balanced_padronized_df['glicose'].replace(to_replace=0, value=median_glicose)

df_teste_balanced_padronized_df['grossura_pele'] = df_teste_balanced_padronized_df['grossura_pele'].replace(to_replace=0, value=median_grossura_pele)

df_teste_balanced_padronized_df['insulina'] = df_teste_balanced_padronized_df['insulina'].replace(to_replace=0, value=median_insulina)
#conferindo valores 0

df_teste_balanced_padronized_df["bmi"].unique()
#retirando os outliers, insulina com valor maximo de 4444

df_teste_balanced_padronized_df["grossura_pele"] = np.where(df_teste_balanced_padronized_df.grossura_pele > 99, 99, df_teste_balanced_padronized_df.grossura_pele)

df_teste_balanced_padronized_df["grossura_pele"].max()
df_teste_balanced_padronized_df["insulina"] = np.where(df_teste_balanced_padronized_df.insulina > 846, 846, df_teste_balanced_padronized_df.insulina)

df_teste_balanced_padronized_df["insulina"].max()
df_teste_balanced_padronized_df["bmi"] = np.where(df_teste_balanced_padronized_df.bmi > 67.1, 67.1, df_teste_balanced_padronized_df.bmi)

df_teste_balanced_padronized_df["bmi"].max()
df_teste_balanced_padronized_df["pressao_sanguinea"] = np.where(df_teste_balanced_padronized_df.pressao_sanguinea > 122, 122, df_teste_balanced_padronized_df.pressao_sanguinea)

df_teste_balanced_padronized_df["pressao_sanguinea"].max()
df_teste_balanced_padronized_df["glicose"] = np.where(df_teste_balanced_padronized_df.glicose > 198, 198, df_teste_balanced_padronized_df.glicose)

df_teste_balanced_padronized_df["glicose"].max()
df_teste_balanced_padronized_df["indice_historico"] = np.where(df_teste_balanced_padronized_df.indice_historico > 2.42, 2.42, df_teste_balanced_padronized_df.indice_historico)

df_teste_balanced_padronized_df["indice_historico"].max()
#verificando as alteraçoes, insulina max 846

df_teste_balanced_padronized_df.describe()
#normalizando os dados , scala

X2 = df_teste_balanced_padronized_df

scaler = StandardScaler().fit(X2)

standardX2 = scaler.transform(X2)

standardX2
#predicao

ynew = model.predict_classes(standardX2)

print("Classe '0': {}".format(sum(ynew==0)))

print("Classe '1': {}".format(sum(ynew==1)))

ynew
#anotaçoes

#xboost melhor resultado seed=629 83,32% - enviar 2 submission com os melhores parametros - ok

#keras overfting ajuste de parametros - enviar as submission com os melhores resultados - ok

#lembrar de postar até segunda - ok

#estudar como diminuir o overting - ok

#validar com matrix confusion pra ver qual classe ta errando mais

#testar xboost com diversos parametros - ok
#testando outros algoritimos

from pandas import read_csv

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# Import dos módulos

from pandas import read_csv

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

df=read_csv('../input/dataset_treino.csv')

i=1

j=50

while i < j:

    median_bmi = df['bmi'].median()

    df['bmi'] = df['bmi'].replace(to_replace=0, value=median_bmi)

    median_pressao_sanguinea = df['pressao_sanguinea'].median()

    df['pressao_sanguinea'] = df['pressao_sanguinea'].replace(to_replace=0, value=median_pressao_sanguinea)

    median_glicose = df['glicose'].median()

    df['glicose'] = df['glicose'].replace(to_replace=0, value=median_glicose)

    median_grossura_pele = df['grossura_pele'].median()

    df['grossura_pele'] = df['grossura_pele'].replace(to_replace=0, value=median_grossura_pele)

    median_insulina = df['insulina'].median()

    df['insulina'] = df['insulina'].replace(to_replace=0, value=median_insulina)

    from sklearn.preprocessing import StandardScaler

    colunas=['num_gestacoes','glicose','pressao_sanguinea','grossura_pele','insulina','bmi','indice_historico','idade']

    balanced_padronized_df=df[colunas]

    X = balanced_padronized_df

    Y = balanced_df.classe

    # Gerando o novo padrão

    scaler = StandardScaler().fit(X)

    standardX = scaler.transform(X)

    X=standardX



    # Definindo o tamanho dos dados de treino e de teste

    teste_size = 0.33

    seed = i

    #print(seed)

    # Criando o dataset de treino e de teste

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size = teste_size, random_state = seed)



    # Criando o modelo

    modelo = XGBClassifier()

    modelo.fit(X_treino, y_treino)

    # Pront do modelo

    #print(modelo)



    # Fazendo previsões

    y_pred = modelo.predict(X_teste)

    previsoes = [round(value) for value in y_pred]



    # Avaliando as previsões

    accuracy = accuracy_score(y_teste, previsoes)

    print("Seed %i Acuracia: %.2f%%" % (i,accuracy * 100.0))

    i=i+1