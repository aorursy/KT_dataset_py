import csv as csv
import pandas as pd
import numpy as np

# le arquivo e separando por ","
bosp = pd.read_csv('bo_sp_2016.csv', delimiter=",")

# ordena arquivo por Ano
bosp.sort_values(by=['ano_bo'], ascending=True)

bosp.head(10000)
# cria lista de objetos chamada classe BoSP
class BoSP(list):
        
     def __init__(self, table):
        self.num_bo = table['num_bo']         
        self.ano_bo = table['ano_bo']
        self.id_delegacia = table['id_delegacia']
        self.nome_departamento =table['nome_departamento']
        self.nome_seccional = table['nome_seccional']
        self.delegacia = table['delegacia']
        self.nome_departamento_circ = table['nome_departamento_circ'] 
        self.nome_seccional_circ = table['nome_seccional_circ'] 
        self.nome_delegacia_circ = table['nome_delegacia_circ']
        self.ano = table['ano']
        self.mes = table['mes']
        self.flag_status = table['flag_status']        
        self.rubrica = table['rubrica'] 
        self.desdobramento = table['desdobramento'] 
        self.conduta = table['conduta'] 
        self.latitude = table['latitude']
        self.longitude = table['longitude']
        self.cidade = table['cidade']
        self.logradouro = table['logradouro']
        self.numero_logradouro = table['numero_logradouro']
        self.flag_status_2 = table['flag_status_2']
        self.location = table['location']
            
        def __str__(self):
            return str(vars(self))
#cria uma intancia da classe 

with open('bo_sp_2016.csv', 'r',encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        bosp1 = [BoSP(c) for c in reader]
        
        for x in bosp1:
            print(x)
#agrupa id delegacia e delegacia

bosp.groupby(['id_delegacia', 'delegacia']).mean()
#cria nova coluna e separa por palavra chave

bosp['roubos'] = bosp['rubrica'].str.partition('Roubo ').mean()
bosp['furtos'] = bosp['rubrica'].str.partition('Furto ').mean()
bosp['lesão'] = bosp['rubrica'].str.partition('Lesão ').mean()

bosp.head(9999)
#filtra tipos de crimes por quantidade - APENAS ANALISE PARA SABER O QUE CONSTA NA COLUNA

bosp.groupby('rubrica').size().nlargest(50)

#cria nova coluna e separa por palavra chave

bosp['furt_veic'] = bosp['conduta'].str.partition('veiculo ').mean()
bosp['roubo_carga'] = bosp['conduta'].str.partition('carga').mean()
#filtra conduta do crimes por quantidade - APENAS ANALISE PARA SABER O QUE CONSTA NA COLUNA
bosp.groupby('conduta').size().nlargest(50)
