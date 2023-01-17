# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Pretendo inferir quantidade de suicídios baseado em sexo, faixa etária, gdp e populacao

df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')



# Observe que o dataset tem colunas redundantes como country-year.

print(df.head())
print(df.columns)

# Geração e country-year não agregam muito, já que sabemos as idades

df = df.drop(columns=['country-year','generation','country','year'],errors='ignore')



print(df.head())

df.rename(columns={'HDI for year':'HDI_for_year',

                   'suicide/100k pop':'suicide_per_100k_pop',

                   ' gdp_for_year ($) ':'gdp_for_year',

                   'gdp_per_capita ($)':'gdp_per_capita'},inplace=True,errors='ignore')

print(df.columns)



print(df.isna().sum())



# aparentemente todos os HDI_for_year da albania estão NaN...

print(df.loc[df['HDI_for_year'].isna()].head())



df_dropna = df.dropna(subset=['HDI_for_year']).copy()
print(df_dropna.columns)

print(df_dropna.iloc[0])



## gdp_for_year usa vírgulas nos números... e tem um espaço antes de seu nome e depois

#https://stackoverflow.com/questions/22137723/convert-number-strings-with-commas-in-pandas-dataframe-to-float#22137890

df_dropna['gdp_for_year'] = df_dropna['gdp_for_year'].str.replace(',','').astype(float)



# tudo ok agora

print(df_dropna.iloc[0])

df_dropna['gdp_per_capita'] = df_dropna['gdp_per_capita'].astype(float)

print(df_dropna.isna().sum())

print(df_dropna.dtypes)



df_dropna_dummies = pd.get_dummies(df_dropna)

print(df_dropna_dummies.head())







X = np.array(df_dropna_dummies.drop(columns=['suicides_no']).copy())

y = np.array(df_dropna_dummies['suicides_no'].copy())





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4)

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train,y_train)

## quanto mais perto de 1, melhor

print("R2 score ",model.score(X_test,y_test))



## mean squared error

from sklearn.metrics import mean_squared_error

print('MSE :',mean_squared_error(y_test,model.predict(X_test)))



print('Erro muito alto. Indica que este modelo não É adequado e grande não-linearidade dos dados')



# Vamos utilizar o mesmo dataset para clusterizar



from sklearn.cluster import KMeans

from sklearn.metrics import v_measure_score





# Minhas labels serão a ordem de grandeza do número de suicídios

# Há as seguintes ordens de grandeza: 0,1,2,3,4 

y = np.array(np.log10(df_dropna_dummies['suicides_no'].copy()+1),dtype=int)



print(set(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.35)



n_clusters = len(set(y))

print('n_clusters:',n_clusters)

model = KMeans(n_clusters = n_clusters, max_iter=300)

model.fit(X_train)



print("o v_measure baixo indica a baixa qualidade do clustering executado")

print(v_measure_score(model.predict(X_test),y_test))



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('Nem o classificador logístico consegue inferir as labels que eu criei artificialmente na validação')

print(model.score(X_test,y_test))

print((y_test==y_pred).sum()/y_test.shape[0])

print('no teste: ', (y_train==model.predict(X_train)).sum()/y_train.shape[0])



print('cenário de underfitting')

#########



### E se tivessemos removido a coluna 'HDI_for_year'?



df_no_HDI = df.drop(columns='HDI_for_year')

df_no_HDI['gdp_for_year'] = df_no_HDI['gdp_for_year'].str.replace(',','').astype(float)

df_no_HDI['gdp_per_capita']=df_no_HDI['gdp_per_capita'].astype(float)

print(df_no_HDI.dtypes)



df_no_HDI = pd.get_dummies(df_no_HDI)



print(df_no_HDI.columns)

print(df_no_HDI.info())

df_dropna_dummies = df_no_HDI.copy()

X = np.array(df_dropna_dummies.drop(columns=['suicides_no']).copy())

y = np.array(df_dropna_dummies['suicides_no'].copy())





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4)

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train,y_train)

## quanto mais perto de 1, melhor

print("R2 score ",model.score(X_test,y_test))



## mean squared error

from sklearn.metrics import mean_squared_error

print('MSE :',mean_squared_error(y_test,model.predict(X_test)))



print('Erro muito alto. Indica que este modelo não É adequado e grande não-linearidade dos dados')


