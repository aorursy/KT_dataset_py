#Embora a base de dados abaixo tenha menos de 250 registros, foi selecionada tento em vista o interesse da unidade. 
#Trata-se da base de dados de viagens internacionais de 2018 na qual podemos analisar dados de gastos com diárias, passagens, afastamentos. 
#Utilizaremos a biblioteca pandas_profiling no caderno
import pandas as pd
import pandas_profiling as pp
import datetime

viagens = pd.read_csv("../input/viagens_internacionais_2018.csv",encoding="UTF-8", sep=";")
viagens.head(5)

# Limpeza e correção de variáveis 

# 1) Converter 'Cotacao dolar','Custo passagem' e 'Custo total viagem' para valores numericos. 
viagens['Cotacao dolar'] = viagens['Cotacao dolar'].str.replace(',','.').astype(float)
viagens['Custo passagem'] = viagens['Custo passagem'].str.replace(',','.').astype(float)
viagens['Custo total viagem'] = viagens['Custo total viagem'].str.replace(',','.').astype(float)
viagens['Valor total diarias'] = viagens['Valor total diarias'].str.replace(',','.').astype(float)
viagens['Valor total passagens'] = viagens['Valor total passagens'].str.replace(',','.').str.replace('-','0').astype(float)

# 2) Remover da listagem os itens Arquivados. Trata-se de eventos que não participamos e, portanto, não têm dados para o profiling
viagens = viagens[viagens['Situação'] != 'Arquivado']

# 3) Após verificar os demais tipos de dados (viagens.dtypes), foi fácil perceber que as datas também careciam de conversão
viagens['Data de início'] = pd.to_datetime(viagens['Data de início'], format='%d/%m/%Y') 
viagens['Data de término'] = pd.to_datetime(viagens['Data de término'], format='%d/%m/%Y')

# Gerar profiling ...
pp.ProfileReport(viagens)


