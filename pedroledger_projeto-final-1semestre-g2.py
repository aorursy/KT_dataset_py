import requests # Responsável por baixar a página web

from bs4 import BeautifulSoup # Responsável por raspar os dados

import pandas as pd # Responsável por tabular os dados obtidos

import re # Permite o uso de expressões regulares

import matplotlib.pyplot as plt # Responsável pelos gráficos de barras e dispersão


dolar = 'https://www.dolarhoje.net.br/dolar-comercial/'

page = requests.get(dolar) # Aqui, fazemos o download da página
page
# Criamos um objeto BeautifulSoup para conseguir raspar os dados
soup = BeautifulSoup(page.content, 'html.parser')

soup.prettify() # Podemos usar o prettify() para exibir o conteúdo da página de maneira mais legível
tabela_geral = soup.find_all('div', class_='e')
tabela_geral # Separamos a tabela para começar a filtrar os dados
janeiro = tabela_geral[1] # Como os dados de cada mês estão em divs diferentes, cada mês recebe um índice de 'tabela_geral'
dias_jan = janeiro.find_all('p') # Encontramos as tags 'p', onde estão os textos desejados
dias_jan.pop(0) # Apagamos o primeiro item, que é somente o nome do respectivo mês e não uma informação útil
dias_jan.pop(0)
dias_jan # Repetimos esse processo para todos os meses

fevereiro = tabela_geral[2]
dias_fev = fevereiro.find_all('p')
dias_fev.pop(0)
dias_fev
marco = tabela_geral[3]
dias_mar = marco.find_all('p')
dias_mar.pop(0)
dias_mar
abril = tabela_geral[4]
dias_abr = abril.find_all('p')
dias_abr.pop(0)
dias_abr
maio = tabela_geral[5]
dias_mai = maio.find_all('p')
dias_mai.pop(0)
dias_mai


junho = tabela_geral[6]
dias_jun = junho.find_all('p')
dias_jun.pop(0)
dias_jun.pop(8)
dias_jun.pop(7)
dias_jun
def prep_linha(linha):
    linha = linha.replace(',', '.') # Troca a vírgula pelo ponto, como separador decimal
    
    linha = linha.replace(' ', '\t') # Substitui os espaços em branco na string por uma tabulação
    
    linha = re.sub('\t-\t', '\t', linha) # Substitui o padrão '\t-\t' por um '\t' apenas
    
    linha = linha.replace('R$', '') # Remove o símbolo que representa a moeda brasileira
    
    linha = linha.replace('%', '') # Remove o símbolo de porcentagem da string
    
    linha = linha.replace('\t\t', '\t')
       
    return linha
texto_final = 'Data:\tPreço:\tVariação(em %):\n'

for i in range(0, len(dias_jan)):
    linha = prep_linha(dias_jan[i].get_text())
    texto_final += linha
    texto_final += '\n'
    
for i in range(0, len(dias_fev)):
    linha = prep_linha(dias_fev[i].get_text())
    texto_final += linha
    texto_final += '\n'
    
for i in range(0, len(dias_mar)):
    linha = prep_linha(dias_mar[i].get_text())
    texto_final += linha
    texto_final += '\n'
    
for i in range(0, len(dias_abr)):
    linha = prep_linha(dias_abr[i].get_text())
    texto_final += linha
    texto_final += '\n'
    
for i in range(0, len(dias_mai)):
    linha = prep_linha(dias_mai[i].get_text())
    texto_final += linha
    texto_final += '\n'
    
for i in range(0, len(dias_jun)):
    linha = prep_linha(dias_jun[i].get_text())
    texto_final += linha
    texto_final += '\n'

print(texto_final)
arquivo = open('variacao-dolar.csv', 'w')
arquivo.write(texto_final)
arquivo.close()
arquivo
df = pd.read_csv('variacao-dolar.csv', sep='\t')
df
df["Preço:"]
x = df["Data:"]
y = df["Preço:"]

plt.plot(x,y, "brown")
plt.title("Variação do Dólar perante a Pândemia do Covid-19 dentre os meses de Janeiro a Junho")

plt.xlabel("None")

plt.ylabel("Dólar-variação")

plt.show()

texto_final_Janeiro = 'Data:\tPreço:\tVariação(em %):\n'

for i in range(0, len(dias_jan)):
    linha = prep_linha(dias_jan[i].get_text())
    texto_final_Janeiro += linha
    texto_final_Janeiro += '\n'

print(texto_final_Janeiro)
arquivo_Janeiro = open('variacao-dolar-Janeiro.csv', 'w')
arquivo_Janeiro.write(texto_final_Janeiro)
arquivo_Janeiro.close()
arquivo_Janeiro 
df_Janeiro= pd.read_csv('variacao-dolar-Janeiro.csv', sep='\t')
df_Janeiro
Seis_dias_de_Janeiro = df_Janeiro.head(6)
Seis_dias_de_Janeiro
Seis_dias_de_Janeiro['Data:']
Seis_dias_de_Janeiro['Preço:']
x = Seis_dias_de_Janeiro['Data:']
y = Seis_dias_de_Janeiro['Preço:']

plt.plot(x,y)


plt.title("Variação do Dólar dentre os 6 primeiros dias de Janeiro em decorrência da Pândemia do Covid-19")

plt.ylabel("Dólar-preço")

plt.xlabel("Data")

plt.show()
texto_final_Fevereiro = 'Data:\tPreço:\tVariação(em %):\n'

for i in range(0, len(dias_fev)):
    linha = prep_linha(dias_fev[i].get_text())
    texto_final_Fevereiro += linha
    texto_final_Fevereiro += '\n'
print(texto_final_Fevereiro)   
arquivo_Fevereiro = open('variacao-dolar-Fevereiro.csv', 'w')
arquivo_Fevereiro.write(texto_final_Fevereiro)
arquivo_Fevereiro.close()
arquivo_Fevereiro
df_Fevereiro = pd.read_csv('variacao-dolar-Fevereiro.csv', sep='\t')
df_Fevereiro
Seis_dias_de_Fevereiro = df_Fevereiro.head(6)
Seis_dias_de_Fevereiro
Seis_dias_de_Fevereiro['Data:']
Seis_dias_de_Fevereiro['Preço:']
x =  Seis_dias_de_Fevereiro['Data:']
y =  Seis_dias_de_Fevereiro['Preço:']


plt.plot(x,y)

plt.title("Variação do Dólar dentre os 6 primeiros dias de Fevereiro em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel('Dólar-variação')

plt.show()
texto_final_Março = 'Data:\tPreço:\tVariação(em %):\n'

for i in range(0, len(dias_mar)):
    linha = prep_linha(dias_mar[i].get_text())
    texto_final_Março += linha
    texto_final_Março += '\n'

print(texto_final_Março)
arquivo_Março = open('variacao-dolar-Março.csv', 'w')
arquivo_Março.write(texto_final_Março)
arquivo_Março.close()
arquivo_Março
df_Março = pd.read_csv('variacao-dolar-Março.csv', sep='\t')
df_Março
Seis_dias_de_Março = df_Março.head(6)
Seis_dias_de_Março
Seis_dias_de_Março['Data:']
Seis_dias_de_Março['Preço:']
x = Seis_dias_de_Março['Data:'] # Aqui colocamos os valores desejaveis dentro de identificadores
y = Seis_dias_de_Março['Preço:'] # Aqui colocamos os valores desejaveis dentro de identificadores

plt.plot(x,y) # Aqui definimos os parâmetros para nosso gráfico, os parâmetros serão(Data, Preço)

plt.title("Variação do Dólar dentre os 6 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel("Dólar-preço")

plt.show()
texto_final_Abril = 'Data:\tPreço:\tVariação(em %):\n'   
    
for i in range(0, len(dias_abr)):
    linha = prep_linha(dias_abr[i].get_text())
    texto_final_Abril += linha
    texto_final_Abril += '\n'
    
print(texto_final_Abril)
    
arquivo_Abril = open('variacao-dolar-Abril.csv', 'w')
arquivo_Abril.write(texto_final_Abril)
arquivo_Abril.close()
arquivo_Abril 
df_Abril = pd.read_csv('variacao-dolar-Abril.csv', sep='\t')
df_Abril
Seis_dias_de_Abril = df_Abril.head(6)
Seis_dias_de_Abril
Seis_dias_de_Abril['Data:']
Seis_dias_de_Abril['Preço:']
x = Seis_dias_de_Abril['Data:']
y = Seis_dias_de_Abril['Preço:']

plt.plot(x,y)

plt.title("Variação do Dólar dentre os 6 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel('Dólar-variação')

plt.show()
texto_final_Maio = 'Data:\tPreço:\tVariação(em %):\n'

for i in range(0, len(dias_mai)):
    linha = prep_linha(dias_mai[i].get_text())
    texto_final_Maio += linha
    texto_final_Maio += '\n'

print(texto_final_Maio)
arquivo_Maio = open('variacao-dolar-Maio.csv', 'w')
arquivo_Maio.write(texto_final_Maio)
arquivo_Maio.close()
arquivo_Maio 
df_Maio = pd.read_csv('variacao-dolar-Maio.csv', sep='\t')
df_Maio
Seis_dias_de_Maio = df_Maio.head(6)
Seis_dias_de_Maio
Seis_dias_de_Maio['Data:']
Seis_dias_de_Maio['Preço:']
y = Seis_dias_de_Maio['Preço:']
x = Seis_dias_de_Maio['Data:']

plt.plot(x,y)

plt.title("Variação do Dólar dentre os 6 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.ylabel("Dólar-preço")

plt.xlabel("Data")

plt.show()
texto_final_Junho = 'Data:\tPreço:\tVariação(em %):\n'

for i in range(0, len(dias_jun)):
    linha = prep_linha(dias_jun[i].get_text())
    texto_final_Junho += linha
    texto_final_Junho += '\n'
print(texto_final_Junho)
    
arquivo_Junho = open('variacao-dolar-Junho.csv', 'w')
arquivo_Junho.write(texto_final_Junho)
arquivo_Junho.close()
arquivo_Junho
df_Junho = pd.read_csv('variacao-dolar-Junho.csv', sep='\t')
df_Junho 
Seis_dias_de_Junho = df_Junho.head(6)
Seis_dias_de_Junho 

Seis_dias_de_Junho["Data:"]
Seis_dias_de_Junho["Preço:"]
x = Seis_dias_de_Junho["Data:"]
y = Seis_dias_de_Junho["Preço:"]

plt.plot(x,y, "orange")

plt.title("Variação do Dólar dentre os 6 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.ylabel("Dólar-preço")

plt.xlabel("Data")

plt.show()
x = cinco_dias_de_Janeiro['Data:']
y = cinco_dias_de_Janeiro['Preço:']

plt.plot(x,y, "purple")


plt.title("Variação do Dólar dentre os 5 primeiros dias de Janeiro em decorrência da Pândemia do Covid-19")


plt.ylabel("Dólar-preço")

plt.xlabel("Data")

plt.show()
x =  cinco_dias_de_Fevereiro['Data:']
y =  cinco_dias_de_Fevereiro['Preço:']


plt.plot(x,y, "blue")

plt.title("Variação do Dólar dentre os 5 primeiros dias de Fevereiro em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel('Dólar-variação')

plt.show()
x = cinco_dias_de_Março['Data:'] # Aqui colocamos os valores desejaveis dentro de identificadores
y = cinco_dias_de_Março['Preço:'] # Aqui colocamos os valores desejaveis dentro de identificadores

plt.plot(x,y, "red") # Aqui definimos os parâmetros para nosso gráfico, os parâmetros serão(Data, Preço)

plt.title("Variação do Dólar dentre os 5 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel("Dólar-preço")

plt.show()
x = cinco_dias_de_Abril['Data:']
y = cinco_dias_de_Abril['Preço:']

plt.plot(x,y, "black")

plt.title("Variação do Dólar dentre os 5 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel('Dólar-variação')

plt.show()
y = cinco_dias_de_Maio['Preço:']
x = cinco_dias_de_Maio['Data:']

plt.plot(x,y, "green")

plt.title("Variação do Dólar dentre os 5 primeiros dias de Maio em decorrência da Pândemia do Covid-19")

plt.ylabel("Dólar-preço")

plt.xlabel("Data")

plt.show()

x =  cinco_dias_de_Fevereiro['Data:']
y =  cinco_dias_de_Fevereiro['Preço:']


plt.plot(x,y, "blue")

plt.title("Variação do Dólar dentre os 5 primeiros dias de Fevereiro em decorrência da Pândemia do Covid-19")

plt.xlabel('Data')

plt.ylabel('Dólar-variação')

plt.show()


y = df["Preço:"]

plt.plot(y, "brown")
plt.title("Variação do Dólar perante a Pândemia do Covid-19 dentre os meses de Janeiro a Junho")

plt.xlabel("None")

plt.ylabel("Dólar-variação")

plt.show()