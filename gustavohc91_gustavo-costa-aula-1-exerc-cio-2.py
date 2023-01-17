import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
import pandas as pd



df = pd.read_csv('../input/anv.csv', delimiter=',')

df
resposta = [["aeronave_operador_categoria", "Qualitativa Nominal"],

            ["aeronave_tipo_veiculo","Qualitativa Nominal"],

            ["aeronave_motor_tipo","Qualitativa Nominal"],

            ["aeronave_motor_quantidade","Quantitativa Discreta"],

            ["aeronave_ano_fabricacao","Quantitativa Continua"], 

            ["aeronave_fase_operacao","Qualitativa Ordinal"], 

            ["total_fatalidades","Quantitativa Discreta"]]



resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
#df.apply(pd.Series.value_counts)



new_df = df[['aeronave_operador_categoria', 'aeronave_tipo_veiculo', 'aeronave_motor_tipo', 'aeronave_fase_operacao']].copy()

df_operador_categ =pd.DataFrame(pd.value_counts(new_df.aeronave_operador_categoria))

df_tipo_veiculo =pd.DataFrame(pd.value_counts(new_df.aeronave_tipo_veiculo))

df_motor_tipo =pd.DataFrame(pd.value_counts(new_df.aeronave_motor_tipo))

df_fase_operacao =pd.DataFrame(pd.value_counts(new_df.aeronave_fase_operacao))



print(df_operador_categ)

print(df_tipo_veiculo)

print(df_motor_tipo)

print(df_fase_operacao)
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.figure(figsize=(10,8))

plt.bar(df_operador_categ.index, df_operador_categ.aeronave_operador_categoria)



plt.xlabel("Categorias")

plt.ylabel("Quantidade por Categoria")

plt.title("Quantidade de Aeronaves por Categoria")

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()

plt.figure(figsize=(10,8))

plt.bar(df_tipo_veiculo.index, df_tipo_veiculo.aeronave_tipo_veiculo, color='green')



plt.xlabel("Tipo")

plt.ylabel("Quantidade por Tipo")

plt.title("Quantidade de Veículos Aéreos por Tipo")

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
plt.figure(figsize=(10,10))

plt.bar(df_motor_tipo.index, df_motor_tipo.aeronave_motor_tipo, color='blue')



plt.xlabel("Tipo")

plt.ylabel("Quantidade por Tipo")

plt.title("Quantidade de Motores de Veículos Aéreos por Tipo")

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.style.use('ggplot')

plt.show()
plt.figure(figsize=(20,10))

plt.bar(df_fase_operacao.index, df_fase_operacao.aeronave_fase_operacao, color='blue')

plt.xlabel("Fase de Opereção")

plt.ylabel("Quantidade por Fase")

plt.title("Quantidade de Operações por Tipo")

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)