import urllib.request

from bs4 import BeautifulSoup

import pandas as pd

import numpy as np # linear algebra

import os

import csv

import seaborn as sns

from sklearn import linear_model

from sklearn import preprocessing



url ='https://imoveis.trovit.com.br/index.php/cod.search_homes/type.2/what_d.cuiab%C3%A1/sug.0/isUserSearch.1/origin.2/order_by.relevance/area_min.30/area_max.60/property_type.Kitnet/'

req = urllib.request.Request(url)



try:

    response = urllib.request.urlopen(req)

except urllib.error.URLError as e:

    print("deu ruim :(")

    

soup = BeautifulSoup(response, 'html.parser')

content = soup.find('div', id='content')

list = content('div', class_='js-item')



f = csv.writer(open('../precosEmCuiaba.csv', 'w'))

f.writerow(['Descricao', 'Tamanho', 'Valor', 'Sobra', 'Periodo_Minimo'])



for item in list:

    properties = item.find(class_='features').find_all('div', class_='property')



    descricao = item.find('div', class_='details').h4.a.text

    tamanho = str(int(properties[-1].text[0:2]))

    valor = float(item.find('div', class_='price').span.text[2:].replace('.',''))

    custoMedio = 998.00 - int(valor)

    periodoMinimo = 12

    #baseando-se no salario minimo

    f.writerow([descricao, tamanho, valor, custoMedio, periodoMinimo])
df = pd.read_csv('../precosEmCuiaba.csv')

df.head()
dfPrecoTamanho = df[['Valor','Tamanho']]

dfPrecoTamanho.head()
sns.pairplot(data=dfPrecoTamanho, kind="reg")

le = preprocessing.LabelEncoder()

regressao = linear_model.LinearRegression()
X = np.array(dfPrecoTamanho['Tamanho']).reshape(-1, 1)

y = le.fit_transform(dfPrecoTamanho['Valor'])

regressao.fit(X, y)
tamanhos = [50,55,60,65,70,75,80,85,90,95]

for i in tamanhos:

    j = regressao.predict(np.array(i).reshape(-1, 1))

    print('Tamanho: ',i,' Valor: ',j,'\n')
custoXPreco = df[['Valor', 'Tamanho', 'Sobra']]

custoXPreco.head()
X = np.array(custoXPreco[['Tamanho', 'Sobra']])

y = le.fit_transform(custoXPreco['Valor'])

regressao.fit(X, y)
tamanho = 32

#m²

salario2016= 788.00-700

#788.00 = salario minimo em 2016

print(' kitinet com tamanho de 32m² em 2016 custaria: R$',int(regressao.predict(np.array([[tamanho, salario2016]]))) * 100)