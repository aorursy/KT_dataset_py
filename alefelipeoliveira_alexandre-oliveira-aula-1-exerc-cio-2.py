import pandas as pd



#Carregando arquivo csv em um dataframe

df = pd.read_csv('../input/anv.csv', delimiter=',')

#Devido a uma melhor representatividade gráfica, foram selecionadas 7 variáveis específicas

#As variáveis escolhidas geram uma representação gráfica de fácil entendimento

var = df.columns[[3,7,8,10,12,13,22]]



#Reduzo o dataframe com só as colunas escolhidas

dataframe = df[var]

dataframe.head(n=1)
#Lista com tipos de variáveis possíveis

var_type = ["Qualitativa Nominal", "Qualitativa Ordinal", "Quantitativa Discreta", "Quantitativa Contínua"]



#Classificando as variáveis

classificacao = [[var[0],var_type[0]], [var[1],var_type[0]], [var[2],var_type[1]], [var[3],var_type[1]], 

            [var[4],var_type[2]], [var[5],var_type[0]], [var[6],var_type[1]]]

classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])

classificacao
#Tabela para tipo de aeronave

tipo = dataframe[var[0]].value_counts().to_frame(name='Frequency')

tipo.head()
#Tabela para tipo de motor da aeronave

motor = dataframe[var[1]].value_counts().to_frame(name='Frequency').head()

motor.head()
#Tabela para quantidade de motores da aeronave

motor_qty = dataframe[var[2]].value_counts().to_frame(name='Frequency')

motor_qty.head()
#Tabela para categoria de peso máximo de decolagem de aeronave

peso_max_decolagem = dataframe[var[3]].value_counts().to_frame(name='Frequency')

peso_max_decolagem.head()
#Tabela para ano de fabricação da aeronave

fabricacao = dataframe[var[4]].value_counts().to_frame(name='Frequency')

fabricacao.head()
#Tabela para país de origem da aeronave

origem = dataframe[var[5]].value_counts().to_frame(name='Frequency')

origem.head()
#Tabela para nível de dano da aeronave por ocorrência

dano = dataframe[var[6]].value_counts().to_frame(name='Frequency')

dano.head()
import matplotlib.pyplot as plt



plt.bar(list(tipo.index), list(tipo['Frequency']))

plt.title('Quantidade relativa ao tipo de aeronave')

plt.ylabel('Quantidade')

labels = list(tipo.index)

plt.xticks(list(tipo.index), labels, rotation='vertical')

plt.xlabel('Tipo')

plt.show()
plt.bar(list(motor.index), list(motor['Frequency']))

plt.title('Quantidade relativa a motor de aeronave')

plt.ylabel('Quantidade')

labels = list(motor.index)

labels[labels.index("***")] = "INDETERMINADA"

plt.xticks(list(motor.index), labels, rotation='vertical')

plt.xlabel('Motor')

plt.show()
plt.bar(list(motor_qty.index), list(motor_qty['Frequency']))

plt.title('Quantidade de motores por aeronave')

plt.ylabel('Quantidade de aeronaves')

labels = list(motor_qty.index)

labels[labels.index("***")] = "INDETERMINADO"

plt.xticks(list(motor_qty.index), labels, rotation='vertical')

plt.xlabel('Quantidade de motores')

plt.show()
labels = list(peso_max_decolagem.index)

labels[labels.index("***")] = "INDETERMINADO"

plt.bar(labels, list(peso_max_decolagem['Frequency']))

plt.title('Quantidade relativa a categoria de peso máximo de decolagem por aeronave')

plt.ylabel('Quantidade')

plt.xlabel('Categoria de Peso máximo de decolagem')

plt.show()
fabricacao = fabricacao.sort_index()



indice = list(fabricacao.index)

frequencia = list(fabricacao['Frequency'])

#Há um valor 0 na lista de anos, que pode comprometer a visualização de dados, será removido

frequencia.pop(indice.index(0))

indice.remove(0)

plt.plot(indice, frequencia)

plt.title('Quantidade de aeronaves produzidas por ano')

plt.ylabel('Quantidade')

plt.xlabel('Ano')

plt.show()
font_size = 25



plt.figure(figsize=(30,10))

plt.subplot(1, 2, 1)

plt.bar(list(origem.index), list(origem['Frequency']))

plt.title('Quantidade relativa a quantidade ao país de origem da aeronave', fontsize = font_size)

plt.ylabel('Quantidade de aeronaves', fontsize = font_size)

labels = list(origem.index)

plt.yticks(fontsize = font_size)

plt.xticks(list(origem.index), labels, rotation='vertical', fontsize = font_size)

plt.xlabel('País de Origem', fontsize = font_size)

plt.subplot(1, 2, 2)

plt.bar(list(origem.index), list(origem['Frequency']))

plt.title('Aeronaves produzidas', fontsize = font_size)

labels = list(origem.index)

plt.yticks(fontsize = font_size)

plt.xticks(list(origem.index), labels, rotation='vertical', fontsize = font_size)

plt.box(False)

plt.suptitle('Utilizando o conceito de Data-Ink Ratio', fontsize = 30)

plt.show()
plt.bar(list(dano.index), list(dano['Frequency']))

plt.title('Quantidade relativa a quantidade ao país de origem da aeronave')

plt.ylabel('Quantidade de aeronaves')

labels = list(dano.index)

plt.xticks(list(dano.index), labels, rotation='vertical')

plt.xlabel('Nível de dano')

plt.show()