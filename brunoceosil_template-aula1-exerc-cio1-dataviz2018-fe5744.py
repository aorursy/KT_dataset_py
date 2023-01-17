import pandas as pd
resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
anv = pd.read_csv('../input/anv.csv')
anv.head(1)
var_anv = [['aeronave_tipo_veiculo', 'Qualitativa Nominal'], ['aeronave_motor_quantidade','Qualitativa Ordinal'], ['aeronave_assentos', 'Quantitativa Discreta'], ['aeronave_ano_fabricacao','Quantitativa Discreta'], ['aeronave_fase_operacao','Qualitativa Nominal'], ['aeronave_tipo_operacao','Qualitativa Nominal'], ['aeronave_nivel_dano','Qualitativa Ordinal'], ['total_fatalidades','Quantitativa Discreta']]
var_anv = pd.DataFrame(var_anv, columns=['Variável', 'Classificação'])
var_anv
anv.head(5)
anv['aeronave_tipo_veiculo'].value_counts()
anv['aeronave_motor_quantidade'].value_counts()
anv['aeronave_fase_operacao'].value_counts()
anv['aeronave_tipo_operacao'].value_counts()
anv['aeronave_nivel_dano'].value_counts()
g1 = anv['aeronave_tipo_veiculo'].value_counts()
gs1 = g1.head(5).plot.bar(title='TOP 5 - ACIDENTES POR TIPO DE AERONAVE')
gs1.set_xlabel('Tipo de Aeronave')
gs1.set_ylabel('Quantidade Acidentes')
gs1.plot()
g2 = anv['aeronave_motor_quantidade'].value_counts()
gs2 = g2.plot.barh(title='ACIDENTES POR TIPO DE MOTOR')
gs2.set_xlabel('Tipo de Motor')
gs2.set_ylabel('Quantidade Acidentes')
gs2.plot()
g3 = anv['aeronave_assentos'].value_counts()
gs3 = g3.head(10).plot.bar(title='TOP 10 - QUANTIDADE DE ASSENTOS EM ACIDENTES')
gs3.set_xlabel('Quantidade de Assentos')
gs3.set_ylabel('Quantidade Acidentes')
gs3.plot()
g4 = anv['aeronave_ano_fabricacao'].value_counts()
gs4 = g4.head(8).plot.bar(title='TOP 8 - ACIDENTES POR ANO DE FABRICAÇÃO DA AERONAVE')
gs4.set_xlabel('Ano de Fabricação')
gs4.set_ylabel('Quantidade Acidentes')
gs4.plot()
g5 = anv['aeronave_fase_operacao'].value_counts()
gs5 = g5.head(7).plot.bar(title='TOP 7 - ACIDENTES POR FASE DE OPERAÇÃO')
gs5.set_xlabel('Fase de Operação')
gs5.set_ylabel('Quantidade Acidentes')
gs5.plot()
g6 = anv['aeronave_tipo_operacao'].value_counts()
gs6 = g6.plot.bar(title='ACIDENTES POR TIPO DE OPERAÇÃO')
gs6.set_xlabel('Fase de Operação')
gs6.set_ylabel('Quantidade Acidentes')
gs6.plot()
g7 = anv['aeronave_nivel_dano'].value_counts()
gs7 = g7.plot.bar(title='NIVEL DE DANOS EM ACIDENTES')
gs7.set_xlabel('Nivel de Dano')
gs7.set_ylabel('Quantidade Acidentes')
gs7.plot()
anv[anv['total_fatalidades'] > 0] = 1
anv[anv['total_fatalidades'] == 0] = 0

g8 = anv['total_fatalidades'].value_counts(normalize=True) * 100
g8.plot.pie(title='ACIDENTES COM FATALIDADES', autopct='%1.1f%%', labels=['Não','Sim'])

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(1)