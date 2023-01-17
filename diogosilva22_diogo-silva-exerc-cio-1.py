import pandas as pd



df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(1)
variaveis = [["aeronave_tipo_veiculo", "Qualitativa Nominal"],

            ["aeronave_modelo","Qualitativa Nominal"],

            ["aeronave_operador_categoria","Qualitativa Nominal"],

            ["aeronave_motor_tipo","Qualitativa Nominal"],

            ["total_fatalidades","Quantitativa Discreta"], 

            ["aeronave_fase_operacao","Qualitativa Discreta"], 

            ["aeronave_ano_fabricacao","Quantitativa Discreta"]]



variaveis = pd.DataFrame(variaveis, columns=["Variavel", "Classificação"])

variaveis


variaveis
tp_veic = pd.DataFrame(df["aeronave_tipo_veiculo"].value_counts())

tp_veic
ae_mod = pd.DataFrame(df["aeronave_modelo"].value_counts())

ae_mod
op_cat = pd.DataFrame(df["aeronave_operador_categoria"].value_counts())

op_cat
f_op = pd.DataFrame(df["aeronave_fase_operacao"].value_counts())

f_op
m_tipo = pd.DataFrame(df["aeronave_motor_tipo"].value_counts())

m_tipo
import matplotlib.pyplot as plt
variaveis
tit = 'Top 5 - Ocorrências por Tipo de Veículo'

df["aeronave_tipo_veiculo"].value_counts().head(5).plot(kind = 'bar', title = tit)
tit = 'Top 3 - Ocorrências por Modelo de Aeronave'

df["aeronave_modelo"].value_counts().head(3).plot(kind = 'barh', title = tit)
tit = 'Top 5 - Ocorrências por Categoria do Operador'

df["aeronave_operador_categoria"].value_counts().head(5).plot(kind = 'bar', title = tit)
tit = 'Top 3 - Ocorrências por Tipo de Motor'

df["aeronave_motor_tipo"].value_counts().head(3).plot(kind = 'barh', title = tit)
tit = 'Top 5 - Ocorrências por Fase de Operação'

df["aeronave_fase_operacao"].value_counts().head(5).plot(kind = 'bar', title = tit)
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(1)