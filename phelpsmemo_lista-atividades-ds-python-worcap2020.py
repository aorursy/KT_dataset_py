# Importação das bibliotecas
from plotnine import * # Biblioteca de visualização de dados
from plotnine.data import diamonds # Conjunto de dados
# Tipo dos dados
type(diamonds)
diamonds
# TODO: Veja os atributos que estão disponíveis nos dados
# Dica: Atributos, em Data Frames são equivalentes as colunas
# pd.DataFrame.columns
# TODO: Visualize as primeiras linhas do conjunto de dados
# pd.DataFrame.head()
# TODO: Visualize a última linha do conjunto de dados
# pd.DataFrame.tail(n = ...)
# TODO: Faça as estatísticas básica dos dados
# pd.DataFrame.describe()
# TODO: Selecione somente as colunas `price` e `carat`
# diamonds[['...', '...']]
# TODO: Selecione todos os atributo exceto `x`, `y` e `z`
# diamonds[diamonds.columns.difference(['...', '...', '...'])]
# TODO: Filtre somente os diamantes que tenham valor maior que 1000 dolares
# diamonds[diamonds['...'] > ...]
# TODO: Filtre somente os diamantes que tenham valor menor que a média dos valores
# diamonds[diamonds['...'] < diamonds['...'].mean()]
# TODO: Filtro somente os diamantes que tenham valor maior que a mediana e que tenham 
# corte igual a `Premium`
# diamonds[(diamonds['...'] > diamonds['...'].median()) & (diamonds['...'] == '...')]
# TODO: Crie um grupo com base no atributo `cut`.
# grupo_1 = diamonds.groupby('...')
# TODO: Utilizando o grupo criado acima, faça a soma dos valores de cada grupo
# grupo_1_soma = grupo_1.`FUNCAO_DE_AGREGACAO`()
# grupo_1_soma['...']
# TODO: Crie um grupo com base no atributo `color`
# grupo_2 = diamonds.groupby('...')
# TODO: No grupo criado acima, calcule a quantidade de elementos em cada grupo
# grupo_2_count = grupo_2.count()
# TODO: Visualize a relação do peso (carat) com o valor (price). Para isto use a geometria de ponto
# ggplot(diamonds, aes(x = '...', y = '...')) + geom_`NOME_DA_GEOMETRIA`()
# TODO: Inserir cores no plot gerado acima, para facilitar a interpretação dos dados
# Os parâmetros `cut` e `clarity` podem ser inseridos como `factor` no parâmetro color, assim, 
# as cores são alteradas

# ggplot(diamonds, aes(x = '...', y = '...', color='factor(...)')) + geom_`NOME_DA_GEOMETRIA`()