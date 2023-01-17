import pandas as pd

cnpj = pd.read_csv('../input/cnpj_dados_cadastrais_pj.csv', sep='#',

                   usecols=['cnpj', 'razao_social', 'cnae_fiscal', 'uf', 'porte_empresa', 'data_inicio_atividade', 'codigo_natureza_juridica', 'situacao_cadastral'])



# Quantidade de Empresas: 40 milhões

cnpj.shape
# Porte da Empresa

cnpj.porte_empresa.value_counts()



# 01 - MICRO EMPRESA

# 05 - DEMAIS

# 03 - EMPRESA DE PEQUENO PORTE

# 00 - NAO INFORMADO
# Status da Empresa - Situação Cadastral

cnpj.situacao_cadastral.value_counts()



# 1 - NULA

# 2 - ATIVA    (18 milhões de empresas ativas)

# 3 - SUSPENSA

# 4 - INAPTA

# 8 - BAIXADA
cnpj.tail()