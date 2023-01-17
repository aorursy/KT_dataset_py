import pandas as pd      #https://pandas.pydata.org
dados = pd.read_csv('../input/aluguel/aluguel.csv',sep=';')
dados
type(dados) #tipo de variável
dados.info()
dados.head(10)
##Informações gerais sobre a base de dados
dados.dtypes
tipos_de_dados = pd.DataFrame(dados.dtypes,columns = ['Tipos de Dados'])
tipos_de_dados.columns.name = 'variáveis' #renomear index
tipos_de_dados
dados.shape #número de linhas , variáveis
dados.shape[0]
dados.shape[1]
#aplicar um valor de código no texto

print('A base de dados apresenta {} registros (imóveis) e {} variáreis'.format(dados.shape[0], dados.shape[1]))
#Ler tabela de site

## pd.read_html('endereço URL')

#caso tenha mais de uma tabela web, utilizar len(tabela) > 1