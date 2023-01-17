import pandas as pd
resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
import pandas as pd
resposta = [["aeronave_tipo_veiculo", "Qualitativa Nominal"],["aeronave_motor_quantidade", "Qualitativa Ordinal"], ["aeronave_pmd_categoria", "Qualitativa Ordinal"], ["aeronave_assentos", "Quantitativa Discreta"], ["aeronave_ano_fabricacao", "Quantitativa Discreta"], ["total_fatalidades", "Quantitativa Discreta"], ["aeronave_nivel_dano", "Qualitativa Ordinal"]]
resposta = pd.DataFrame(resposta, columns = ["Variável", "Classificação"])
resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')

# aeronave_tipo_veiculo
aeronave_tipo_veiculo_count = df['aeronave_tipo_veiculo'].value_counts()
aeronave_tipo_veiculo_porc = df['aeronave_tipo_veiculo'].value_counts(normalize=True) * 100
resultado_aeronave_tipo_veiculo = pd.concat([aeronave_tipo_veiculo_count, aeronave_tipo_veiculo_porc], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa (%)'])
resultado_aeronave_tipo_veiculo

# aeronave_motor_quantidade
aeronave_motor_quantidade_count = df['aeronave_motor_quantidade'].value_counts()
aeronave_motor_quantidade_porc = df['aeronave_motor_quantidade'].value_counts(normalize=True) * 100
resultado_aeronave_motor_quantidade = pd.concat([aeronave_motor_quantidade_count, aeronave_motor_quantidade_porc], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa (%)'])
resultado_aeronave_motor_quantidade

# aeronave_pmd_categoria
aeronave_pmd_categoria_count = df['aeronave_pmd_categoria'].value_counts()
aeronave_pmd_categoria_porc = df['aeronave_pmd_categoria'].value_counts(normalize=True) * 100
resultado_aeronave_pmd_categoria = pd.concat([aeronave_pmd_categoria_count, aeronave_pmd_categoria_porc], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa (%)'])
resultado_aeronave_pmd_categoria

# aeronave_nivel_dano
aeronave_nivel_dano_count = df['aeronave_nivel_dano'].value_counts()
aeronave_nivel_dano_porc = df['aeronave_nivel_dano'].value_counts(normalize=True) * 100
resultado_aeronave_nivel_dano = pd.concat([aeronave_nivel_dano_count, aeronave_nivel_dano_porc], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa (%)'])
resultado_aeronave_nivel_dano



#Fatalidades por tipo de veiculo
#Neste gráficos podemos analisar em quais tipos de veículo mais pessoas morreram.
import matplotlib.pyplot as plt

%matplotlib inline
x = df['aeronave_tipo_veiculo']
y = df['total_fatalidades']


# Criando um gráfico
plt.title('Fatalidades por tipo de veiculo')
plt.xticks(rotation='vertical')
plt.xlabel('Tipo de veículo')
plt.ylabel('Quantidade de fatalidades')
grafico = plt.bar(x, y)
plt.show()
# Quantidade de aeronaves fabricadas por ano
# Neste gráfico podemos observar os 10 anos que mais tiveram veículos fabricados.
g2 = df['aeronave_ano_fabricacao'].value_counts()
g2 = g2.head(10)
gs2 = g2.plot.bar(title='10 anos que mais fabricaram veículos', color='b')
gs2.set_xlabel('Ano de fabricação')
gs2.set_ylabel('Quantidade fabricada')
gs2.plot()

# Quantidade de fatalidades em acidentes de nivel dano leve
%matplotlib inline
x = df['aeronave_nivel_dano']
y = df['total_fatalidades']

# Criando um gráfico
plt.title('Fatalidades por nível dano')
plt.xticks(rotation='vertical')
plt.xlabel('Nível dano')
plt.ylabel('Fatalidades')
grafico = plt.bar(x, y)
plt.show()
# Quantidade de fatalidade em cada nível de dano
df.groupby('aeronave_nivel_dano')['total_fatalidades'].nunique().nlargest(3).plot(kind='pie',legend=False)
plt.title('Os 3 danos da aeronave que mais tiveram fatalidades')
plt.show()
pd.show_versions ()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)