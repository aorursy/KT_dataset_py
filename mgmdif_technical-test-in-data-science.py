import json 

import pandas as pd 



#load json object

with open('../input/realties.json') as f:

    d = json.load(f)



realties_df = pd.DataFrame(d)
cols = ['_source.error','_source.uf', '_source.longitude', '_source.url', '_source.geometry', '_source.condominio', '_source.cidade_uf', '_source.scraping',

'_source.default','_source.area_util','_source.area_total','_source.contato_imobiliaria','_source.valor','_source.isportal','_source.isparticular',

'_source.nome_corretor','_source.banheiro','_source.tipo_negocio','_source.quarto','_source.status','_source.title','_source.opcionais','_source.latitude',

'_source.cidade','_source.descricao','_source.lista_fotos','_source.bairro','_source.garagem','_source.iptu','_source.data_inclusao','_source.codigo',

'_source.finalidade','_source.geohash','_source.tipo_imovel','_source.zoneamento','_source.anunciante','_source.endereco','_source.area_construida','_source.suite',

'_source.area_terreno','_source.area_privativa','_source.cep','_source.numero_corretor','_source.idade_imovel','_source.edificio','_source.isconstrutora',

'_source.area_comum','_source.seguro_incendio','_source.email_corretor','_source.nome_contato','_source.numero_contato_imagem','_source.complemento','_source.terreno_fundo',

'_source.terreno_frente','_source.terreno_esquerda','_source.testada']

source_df = pd.DataFrame(d['_source']).T

source_df.columns = cols
cols = ['_source.zoneamento.id','_source.zoneamento.sigla','_source.zoneamento.nome']

zoneamento_json = source_df['_source.zoneamento'].to_dict()

zoneamento_df = pd.DataFrame(zoneamento_json).T

zoneamento_df.columns = cols
cols = ['_source.geometry.type','_source.geometry.coordinates']

geometry_json = source_df['_source.geometry'].to_dict()

geometry_df = pd.DataFrame(geometry_json).T

geometry_df.columns = cols
cols = ['_source.cidade.short_name','_source.cidade.full_name']

cidade_json = source_df['_source.cidade'].to_dict()

cidade_df = pd.DataFrame(cidade_json).T

cidade_df.columns = cols
cols = ['_source.bairro.id','_source.bairro.nome']

bairro_json = source_df['_source.bairro'].to_dict()

bairro_df = pd.DataFrame(bairro_json).T

bairro_df.columns = cols
cols = ['_source.anunciante.id','_source.anunciante.nome']

anunciante_json = source_df['_source.anunciante'].to_dict()

anunciante_df = pd.DataFrame(anunciante_json).T

anunciante_df.columns = cols
cols = ['_source.scraping.spider_name','_source.scraping.padrao']

scraping_json = source_df['_source.scraping'].to_dict()

scraping_df = pd.DataFrame(scraping_json).T

scraping_df.columns = cols

scraping_df.head()
source_df = source_df.drop(columns=['_source.zoneamento','_source.geometry','_source.cidade','_source.bairro','_source.anunciante','_source.scraping'])

realties_df = realties_df.drop(columns=['_source'])

result = pd.concat([realties_df,source_df,zoneamento_df,geometry_df,cidade_df,bairro_df,anunciante_df,scraping_df], axis=1, sort=False)
result.head()
from scipy import stats

import numpy as np



# Dividing features into numerical and categorical 

num_attr = result.select_dtypes(include=["number"])

cat_attr = result.select_dtypes(exclude=["number"])
# Fill number values with median so cant damage stats calculations

num_attr.fillna(num_attr.median(), inplace=True)
# Fill values != number types with NaN

cat_attr.fillna('missing',inplace=True)
Q1 = num_attr.quantile(0.02)

Q3 = num_attr.quantile(0.98)

IQR = Q3 - Q1

idx = ~((num_attr < (Q1 - 1.5 * IQR)) | (num_attr > (Q3 + 1.5 * IQR))).any(axis=1)
result_cleaned = pd.concat([num_attr.loc[idx], cat_attr.loc[idx]], axis=1)

result_cleaned.head()