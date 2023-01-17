import pandas as pd

pd.options.mode.chained_assignment = None



import plotly.express as px



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
# Importando os dados

dados = pd.read_csv('/kaggle/input/iris/Iris.csv')



# Exibindo as 5 primeiras entradas do nosso conjunto de dados

dados.head(5)
# Verificando a descrição dos dados

dados.describe()
# Criando box plot referente ao comprimento da sépala

fig_comprimento_sepala = px.box(dados, x='Species', y='SepalLengthCm', color='Species')



# Inserindo títulos aos eixos do gráfico

fig_comprimento_sepala.update_layout(xaxis_title='Espécies', 

                                     yaxis_title='Comprimento da sépala')



# Exbibindo o box plot

fig_comprimento_sepala.show()
# Criando o box plot referente a largura da sépala

fig_largura_sepala = px.box(dados, x='Species', y='SepalWidthCm', color='Species')



# Inserindo títulos aos eixos do gráfico

fig_largura_sepala.update_layout(xaxis_title='Espécies', 

                                 yaxis_title='Largura da sépala')



# Exibindo o box plot

fig_largura_sepala.show()
# Criando gráfico de pontos - Comprimento x Largura

fig_caracteristicas_sepala = px.scatter(dados, x="SepalLengthCm", y="SepalWidthCm", 

                                        color="Species")



# Inserindo títulos aos eixos do gráfico

fig_caracteristicas_sepala.update_layout(xaxis_title='Comprimento da sépala', 

                                         yaxis_title='Largura da sépala')



# Exibindo o gráfico de pontos

fig_caracteristicas_sepala.show()
# Criando box plot referente ao comprimento da pétala

fig_comprimento_petala = px.box(dados, x='Species', y='PetalLengthCm', color='Species')



# Inserindo títulos aos eixos do gráfico

fig_comprimento_petala.update_layout(xaxis_title='Espécies', 

                                     yaxis_title='Comprimento da pétala')



# Exbibindo o box plot

fig_comprimento_petala.show()
# Criando box plot referente a largura da pétala

fig_largura_petala = px.box(dados, x='Species', y='PetalWidthCm', color='Species')



# Inserindo títulos aos eixos do gráfico

fig_largura_petala.update_layout(xaxis_title='Espécies', 

                                 yaxis_title='Largura da pétala')



# Exbibindo o box plot

fig_largura_petala.show()
# Criando gráfico de pontos - Comprimento x Largura

fig_caracteristicas_petala = px.scatter(dados, x="PetalLengthCm", y="PetalWidthCm", 

                                        color="Species")



# Inserindo títulos aos eixos do gráfico

fig_caracteristicas_petala.update_layout(xaxis_title='Comprimento da pétala', 

                                         yaxis_title='Largura da pétala')



# Exibindo o gráfico de pontos

fig_caracteristicas_petala.show()
# Criando um DataFrame com a mediana de todas as características por espécie

data_frame_caracteristicas = dados.groupby(['Species'], as_index=False)[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].median()



# Exbindo o novo DataFrame criado

data_frame_caracteristicas
# Definindo as características

X = dados.drop(["Species", 'Id'], axis=1)



# Definindo os rótulos

y = dados['Species']
# Separando os dados em treino e teste 

# 30% dos dados serão para teste

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)
# Criando o modelo Naive Bayes

modelo_nb = GaussianNB()



# Treinando o modelo

modelo_nb.fit(X_treino, y_treino)
# Realizando predições

predicao = modelo_nb.predict(X_teste)
# Porcentagem de acertos

precisao = accuracy_score(y_teste, predicao)



# Exibindo precisão do modelo - Porcentagem de acertos

print("Precisão do modelo: %.2f%%" %(precisao*100))
# Número de acertos

num_acertos = accuracy_score(y_teste, predicao, normalize=False)



# Exibindo número de previsões

print("Número de previsões:", predicao.size)



# Exibindo precisão do modelo - Número de acertos

print("Número de acertos do modelo:", num_acertos)
# Criando um DataFrame para exibir a comparação

df_comparacao = X_teste



# Criando coluna no DataFrame para os rótulos reais

df_comparacao['Rótulo real'] = y_teste



# Criando coluna no DataFrame para os rótulos preditos

df_comparacao['Rótulo predito'] = predicao.tolist()



# Exibindo os 10 primeiras entradas do nosso novo conjunto

df_comparacao.head(10)