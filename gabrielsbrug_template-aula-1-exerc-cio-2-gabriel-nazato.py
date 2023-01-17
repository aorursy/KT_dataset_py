import pandas as pd

import matplotlib.pyplot as plt



resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(3)
# Cria-se lista de colunas escolhidas e qual sua classificação:

resposta = [["aeronave_operador_categoria","Qualitativa Nominal"],["aeronave_tipo_veiculo","Qualitativa Nominal"],["aeronave_motor_tipo","Qualitativa Nominal"],

            ["aeronave_motor_quantidade","Qualitativa Nominal"],["aeronave_pmd_categoria","Qualitativa Nominal"],["aeronave_assentos","Quantitativa Discreta"],

            ["total_fatalidades","Quantitativa Discreta"]]

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
# df somente com as colunas escolhidas

df[resposta["Variavel"]].head(1)
# Tabela de Frequência para a coluna "aeronave_operador_categoria"

df["aeronave_operador_categoria"].value_counts()
# Tabela de Frequência para a coluna "aeronave_tipo_veiculo"

df["aeronave_tipo_veiculo"].value_counts()
# Tabela de Frequência para a coluna "aeronave_motor_tipo"

df["aeronave_motor_tipo"].value_counts()
# Tabela de Frequência para a coluna "aeronave_motor_quantidade"

df["aeronave_motor_quantidade"].value_counts()
# Tabela de Frequência para a coluna "aeronave_pmd_categoria"

df["aeronave_pmd_categoria"].value_counts()
# df somente com as colunas escolhidas

df[resposta["Variavel"]].head(1)
graph = df["aeronave_operador_categoria"].value_counts().plot(kind='bar')

graph.set_title("aeronave_operador_categoria", fontsize=18)

graph.set_ylabel("qty", fontsize=12);
graph = df["aeronave_tipo_veiculo"].value_counts().plot(kind='bar')

graph.set_title("aeronave_tipo_veiculo", fontsize=18)

graph.set_ylabel("qty", fontsize=12);
graph = df["aeronave_motor_tipo"].value_counts().plot(kind='bar')

graph.set_title("aeronave_motor_tipo", fontsize=18)

graph.set_ylabel("qty", fontsize=12);
graph = df["aeronave_motor_quantidade"].value_counts().plot(kind='bar')

graph.set_title("aeronave_motor_quantidade", fontsize=18)

graph.set_ylabel("qty", fontsize=12);
graph = df["aeronave_pmd_categoria"].value_counts().plot(kind='bar')

graph.set_title("aeronave_pmd_categoria", fontsize=18)

graph.set_ylabel("qty", fontsize=12);
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)