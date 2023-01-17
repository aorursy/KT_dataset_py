# importazione delle librerie

import pandas as pd

import pandas_profiling

import numpy as np



from statsmodels.formula.api import ols

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="ticks")



from ml_metrics import rmse



from catboost import CatBoostRegressor 
df = pd.read_csv('../input/price-of-flats-in-moscow/flats_moscow.csv')
df.head()
df = df.drop(['Unnamed: 0'], axis=1)  # rimuovo la colonna 'Unnamed: 0'
# trasformazione delle variabili categoriche in 'category' 

df["walk"] = df["walk"].astype('category')

df["brick"] = df["brick"].astype('category')

df["floor"] = df["floor"].astype('category')

df["code"] = df["code"].astype('category')
# classi delle diverse colonne che compongono il dataset

print(df.dtypes)
print(df['price'].describe())

plt.figure()

sns.distplot(df['price'], color='g', bins=100, hist_kws={'alpha': 0.4})

plt.title('Distribuzione della variabile price - variabile risposta; Prezzo in dollari.');
print(df['totsp'].describe())

plt.figure()

sns.distplot(df['totsp'], color='g', bins=100, hist_kws={'alpha': 0.4})

plt.title('Distribuzione della variabile totsp - predittore; metri quadrati.');
print(df['livesp'].describe())

plt.figure()

sns.distplot(df['livesp'], color='g', bins=100, hist_kws={'alpha': 0.4})

plt.title('Distribuzione della variabile livesp - predittore; metri quadrati calpestabili.');
print(df['kitsp'].describe())

plt.figure()

sns.distplot(df['kitsp'], color='g', bins=100, hist_kws={'alpha': 0.4})

plt.title('Distribuzione della variabile kitsp - predittore; metri quadrati cucina.');
print(df['dist'].describe())

plt.figure()

sns.distplot(df['dist'], color='g', bins=100, hist_kws={'alpha': 0.4})

plt.title('Distribuzione della variabile dist - predittore; distanza dal centro in km.');
print(df['metrdist'].describe())

plt.figure()

sns.distplot(df['metrdist'], color='g', bins=100, hist_kws={'alpha': 0.4})

plt.title('Distribuzione della variabile metrdist - predittore; distanza dalla più vicina stazione della metropolitana in minuti.');
corr = df.select_dtypes(exclude=['category']).corr() 





plt.figure(figsize=(12, 10))

sns.heatmap(corr, 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True)

plt.title('Matrice di correlazione fra le vraiabili numeriche del dataframe.');
# matrice di scatter plot con il dataframe da cui vengono escluse le variabili categoriche 'category'.

sns.pairplot( # matrice di scatter plot con seaborn importato come sns

    df.select_dtypes(exclude=['category']) # df da cui sono state escluse le colonne 'category'; annidare il 

                                           # comando consente di non creare un nuovo dataframe

)
# barplot della variabile Walk (category)

plt.figure()

sns.barplot(x=df.walk.value_counts().index, y=df.walk.value_counts())

plt.title('Conteggio delle case vicine e lontane rispetto alle stazioni della metropolitana - WALK.')

plt.ylabel('Frequenza', fontsize=12)

plt.xlabel('Brick', fontsize=12)

plt.show()
# barplot della variabile Brick (category)

plt.figure()

sns.barplot(x=df.brick.value_counts().index, y=df.brick.value_counts())

plt.title('Frequenze delle case in mattoni o in materiali diversi - BRICK.')

plt.ylabel('Frequenza', fontsize=12)

plt.xlabel('Brick', fontsize=12)

plt.show()
# barplot della variabile Floor (category)

plt.figure()

sns.barplot(x=df.floor.value_counts().index, y=df.floor.value_counts())

plt.title('Conteggio delle case al piano terra o a piani superiori - FLOOR.')

plt.ylabel('Frequenza', fontsize=12)

plt.xlabel('Floor', fontsize=12)

plt.show()
# barplot della variabile Code (category)

plt.figure()

sns.barplot(x=df.code.value_counts().index, y=df.code.value_counts())

plt.title('Conteggio delle diverse zone - CODE.')

plt.ylabel('Frequenza', fontsize=12)

plt.xlabel('Code', fontsize=12)

plt.show()
# modello da stimare - regressione lineare OLS

RegressioneLineare = ols("price ~ totsp + livesp + kitsp + dist + metrdist + walk + brick + floor + code", data=df)
# stima del modello

Risultati = RegressioneLineare.fit()
# stampo i risultati della stima del modello. NB: omettendo il print() i risultati vengono visualizzati lo stesso ma senza formattazione

print(RegressioneLineare.fit().summary())
PuntiTeoriciRegLin = Risultati.predict(df.drop(['price'], axis=1)) # calcolo i punti teorici
RmseRegressione = rmse(actual = df.price, predicted = PuntiTeoriciRegLin) # calcolo dell'indice di Errore RMSE 
# stampo il valore RMSE della regressione lineare # 28.12

print(round(RmseRegressione, 2))
train_data = df.drop(['price'], axis=1)

train_labels = df.price

cat_dims = df.select_dtypes(include=['category'])
mod_cat01 = CatBoostRegressor(loss_function = 'RMSE',

                              iterations = 1450,

                              depth = 16,

                              random_seed = 5,

                              task_type = "GPU",

                              devices = '0:1')
mod_cat01.fit(train_data,

          train_labels,

          cat_features=cat_dims,

          verbose = False)
fea_imp = pd.DataFrame({'imp': mod_cat01.feature_importances_, 'col': train_data.columns})

fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)

plt.title('Modello CatBoost 01 - Feature Importance')

plt.ylabel('Variabile')

plt.xlabel('Importance');
PuntiTeoriciCatboost01 = mod_cat01.predict(train_data)
RmseCatboost01 = rmse(actual = df.price, predicted = PuntiTeoriciCatboost01)
print(round(RmseCatboost01, 2))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharey=True, dpi=120)



ax1.scatter(df.price, PuntiTeoriciRegLin)  # greendots

ax2.scatter(df.price, PuntiTeoriciCatboost01)  # bluestart



ax1.set_title('Modello Regressione OLS - Previsti e osservati'); ax2.set_title('Modello CatBoost 01 - Previsti e osservati')

ax1.set_xlabel('Osservati');  ax2.set_xlabel('Osservati')  # x label

ax1.set_ylabel('Previsiti - OLS');  ax2.set_ylabel('Previsiti - Catboost')  # y label



ax1.plot([df.price.min(), df.price.max()], [df.price.min(), df.price.max()], 'k--', lw=4)

ax2.plot([df.price.min(), df.price.max()], [df.price.min(), df.price.max()], 'k--', lw=4)



ax1.annotate(f"RMSE:  {round(RmseRegressione, 2)}", xy=(100, 700),  xytext=(100, 700))

ax2.annotate(f"RMSE:  {round(RmseCatboost01, 2)}", xy=(100, 700),  xytext=(100, 700))



plt.tight_layout()

plt.show()