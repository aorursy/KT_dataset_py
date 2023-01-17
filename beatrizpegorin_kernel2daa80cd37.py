url_pagina_html='https://g1.globo.com/bemestar/coronavirus/noticia/2020/03/22/casos-de-coronavirus-no-brasil-em-22-de-marco.ghtml'
# Responsavel por se comunicar com o servidor e fazer o download da pagina

import requests



# Responsavel por extrair (raspagem) as informações da página que se fez o download

from bs4 import BeautifulSoup



# Responsavel por analisar as informações da página

import pandas as pd
pagina_html = requests.get(url_pagina_html)





#if pagina_html.status_code == 200:

#o download da pagina foi feito com sucesso



# Mostra o código de retorno da página

print(pagina_html)
#exibe o conteudo da pagina

print(pagina_html.content)
soup = BeautifulSoup(pagina_html.content, 'html.parser')

lista_de_tabelas = soup.find_all('table')

lista_de_tabelas
lista_de_tabelas = soup.find_all('table')[0].get_text()

lista_de_tabelas
lista_de_listas = lista_de_tabelas.split()

lista_de_listas
lista_de_listas.remove('Ministério')

lista_de_listas.remove('da')

lista_de_listas.remove('da')

lista_de_listas.remove('Estado')

lista_de_listas.remove('Secretarias')

lista_de_listas.remove('Saúde')

lista_de_listas.remove('Saúde')

lista_de_listas.remove('Total')

lista_de_listas.remove('1604')

lista_de_listas.remove('1546')

lista_de_listas
def quebrar(lista_de_listas, n):

    inicio = 0

    for i in range(n):

        final = inicio + len(lista_de_listas[i::n])

        yield lista_de_listas[inicio:final]

        inicio = final

#Para agrupa-las de 3 em 3 a lista deveria ser divida em 28 "pedaços"

l = lista_de_listas

obj=(list(quebrar(l, 27))) 

obj
dfObj = pd.DataFrame(obj, columns = ['Estados' , 'Secretaria_da_Saúde', 'Ministério_da_Saúde'])

print(dfObj)
#converte os valores de string para float para que seja possível calcular a média

dfObj = dfObj.astype({"Ministério_da_Saúde": int})

print(dfObj['Ministério_da_Saúde'].mean())
dfObj = dfObj.astype({"Secretaria_da_Saúde": int})

print(dfObj['Secretaria_da_Saúde'].mean())
print(dfObj.sort_values('Secretaria_da_Saúde'))
#Importando a biblioteca matplotlib para visualizaçao de dados

import matplotlib.pyplot as plt

%matplotlib inline
#PARA CONTAR O NUMERO DE DIGITOS DOS CASOS DA SECRETARIA DA SAUDE

dfObj = dfObj.astype({"Secretaria_da_Saúde": str})

dfObj['Secretaria_da_Saúde'].str.len()
#AGORA VAMOS INSERIR UMA COLUNA "TAM" NO DATAFRAME

dfObj['tam']=dfObj['Secretaria_da_Saúde'].str.len()

dfObj
#CLASSIFICANDO EM ORDEM DECRESCENTE DE ACORDO COM O TAMANHO

dfObj.sort_values(by=['tam'], ascending=False)
#MEDIA DE DIGITOS DO NUMERO DE CASOS

dfObj.tam.mean()
#PLOTAREMOS O GRAFICO DO MINISTERIO DA SAUDE E DA SECRETARIA E OBSERVAREMOS AS DIFERENÇAS

dfObj = dfObj.astype({"Secretaria_da_Saúde": int})

ax = plt.gca()



dfObj.plot(kind='bar',x='Estados',y='Secretaria_da_Saúde',ax=ax)

dfObj.plot(kind='bar',x='Estados',y='Ministério_da_Saúde', color='red', ax=ax)



plt.show()
#PLOTANDO UM GRAFICO EM LINHAS PARA MELHOR VISUALIZAÇAO 

ax = plt.gca()





dfObj.plot(kind='line',x='Estados',y='Ministério_da_Saúde',ax=ax)

dfObj.plot(kind='line',x='Estados',y='Secretaria_da_Saúde', color='red', ax=ax)



plt.show()
#AGORA PEGAREMOS OS VALORES MAIORES OU IGUAIS A 100

mais_q100= dfObj.query('tam > 2').head(10)

mais_q100
#PLOTANDO O GRAFICO DOS CASOS MAIORES OU IGUAIS A 100

mais_q100.plot(kind='bar',x='Estados',y='Ministério_da_Saúde')

plt.show()
#OBTENDO VALORES MAIORES OU IGUAIS A 10

mais_q10= dfObj.query('tam == 2').head(12)

mais_q10
#PLOTANDO O GRAFICO DOS VALORES MAIORES OU IGUAIS A 10

mais_q10.plot(kind='bar',x='Estados',y='Ministério_da_Saúde')

plt.show()
#PARA OBTER VALORES MENORES DO QUE 10

menos_q10 = dfObj.query('tam < 2').head(16)

menos_q10
#PLOTANDO GRAFICO DE VALORES MENORES DO Q 10

menos_q10.plot(kind='bar',x='Estados',y='Ministério_da_Saúde')

plt.show()
#CONTANDO QUANTOS ESTADOS TEM 1 DIGITO, 2 DIGITOS OU 3 DIGITOS

dfObj.groupby(by='tam').size()
#O VALOR MEDIO DE CADA GRUPO (1/2/3 DIGITO(S)) DE ACORDO COM A SECRETARIA DA SAUDE

dfObj.groupby(by='tam')['Secretaria_da_Saúde'].mean()
#O VALOR MEDIO DE CADA GRUPO (1/2/3 DIGITO(S)) DE ACORDO COM O MINISTERIO DA SAUDE

dfObj.groupby(by='tam')['Ministério_da_Saúde'].mean()
# A QUANTIDADE DE ESTADOS/LINHAS

total= len(dfObj)

total
#FREQUENCIA DE CASOS COM 1/2/3 DIGITOS

dfObj.groupby('tam').Secretaria_da_Saúde.count().divide(len(dfObj))
#PLOTANDO GRAFICO DE FREQUENCIA

graf_freq=dfObj.groupby('tam').Secretaria_da_Saúde.count().divide(len(dfObj))

graf_freq.plot(kind="bar", figsize=(6,4))

plt.show()