import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
data = pd.read_json('../input/propositions_det_1998_2018.json')
deputados = pd.read_csv('../input/deputies_1998_2018.csv')
"""
    PL
""" 
idtipos = [139]
# filtra por tipo de Autor, neste caso deputado, seleciona as uri's e remove o id de cada uri 
data['dados.id'] = data[data['dados.tipoAutor'] == "Deputado"]['dados.uriAutores'].apply(lambda t: int(t[t.rindex("/") + 1:]) if not t == None else None)
#filtra pelos tipos escolhidos de proposições
data = data[data['dados.idTipo'].isin(idtipos)]
#remove colunas vazias
data.dropna(how='all',axis=1,inplace=True)
#remove os deputados onde o id não estava na API
deputados = deputados[deputados['dados.id'].notnull()]
deputados['dados.id'] = deputados['dados.id'].astype('int32')
#remove as proposições orfãs de autores deputados
data = data[data['dados.id'].isin(deputados.set_index('dados.id',verify_integrity=True).index.tolist())]
#realiza um join onde junta a quantidade total de deputados com suas respectivas quantidades de proposições
deputados = deputados.set_index('dados.id').join(data['dados.id'].value_counts()).rename({'dados.id':'quantProp'},axis='columns')
tops = deputados[['dados.nomeCivil','quantProp']]
tops.sort_values(by='quantProp',ascending=False).head(50)
tops['quantProp'].notnull().value_counts()
tops['quantProp'].sum()
tops.hist(bins=100)
tops.boxplot(vert=False,figsize=(15,5),grid=False)
