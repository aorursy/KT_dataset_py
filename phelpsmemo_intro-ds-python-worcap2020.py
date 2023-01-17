import pandas as pd
# ToDo: Carregar o conjunto de dados
# dados_queimadas = pd.read_csv(..., encoding = 'latin')
# ToDo: Listas os 5 primeiros elementos da tabela carregada
# dados_queimadas.head(n = ...)
dados_queimadas["year"].unique()
# ToDo: Visualizar os valores únicos do atributo `state`
# dados_queimadas[...].unique()
# ToDo: Visualizar os valores únicos do atributo `month`
# dados_queimadas[...].unique()
# ToDo: Visualizar os valores únicos do atributo `date`
# dados_queimadas[...].unique()
# ToDo: Visualizar os valores únicos do atributo `number`
# dados_queimadas[...].unique()
# ToDo: Filtrar os dados utilizando índice lógico
# dados_queimadas_amazonas = dados_queimadas[dados_queimadas["..."] == ...]
# ToDo: Liste os 3 primeiros itens
# dados_queimadas_amazonas.head(n = ...)
# ToDo: Agrupe os dados utilizando o atributo `year`
# dados_queimadas_amazonas_por_ano = dados_queimadas_amazonas.[ ... ]("year")
# ToDo: Aplique uma agregação de soma nos dados que foram agrupados pelo atributo `year`
# dados_queimadas_amazonas_por_ano = dados_queimadas_amazonas_por_ano.[ ... ]
# ToDo: Veja o conteúdo do atributo `number`
# dados_queimadas_amazonas_por_ano["..."]
from plotnine import *
# ToDo: Faça o plot da variação da identificação de fogos no Amazonas utilizando plotnine
# Lembre-se de que, o atributo `number` (y) representa a quantidade de identificações e o índice do pandas.DataFrame representa os anos
# assim, há uma relação ano -> number no pandas.DataFrame

# (
#     ggplot(dados_queimadas_amazonas_por_ano, aes(x = "dados_queimadas_amazonas_por_ano.index", y = "..."))
#         + geom_line()
#         + ggtitle("Variação da identificação de fogos na floresta no Amazonas (1998 à 2017)")
#         + xlab("Ano")
#         + ylab("Quantidade de identificações")
# )
# ToDo: Faça o plot do atributo `number`
# dados_queimadas_amazonas_por_ano["..."].plot(figsize = (8, 8))
# ToDo: Aplicar um filtro lógico para identificar o ano que possui o valor máximo
# dados_queimadas_amazonas_por_ano[dados_queimadas_amazonas_por_ano["..."] == dados_queimadas_amazonas_por_ano["..."].max()]
# ToDo: Com filtro lógico, selecione os dados do Amazonas para encontrar as identificações de incêndios no ano de 2002
# dados_queimadas_amazonas_2002 = dados_queimadas_amazonas[ ... ]
# dados_queimadas_amazonas_2002
# ToDo: Filtre para encontrar o mês com a maior quantidade de identificações em 2002, no Amazonas.
# dados_queimadas_amazonas_2002[dados_queimadas_amazonas_2002["number"] == dados_queimadas_amazonas_2002["number"].[ ... ]]
# ToDo: Agrupar pelos atributos `year` e `state`
# dados_queimadas_soma_dos_anos = dados_queimadas.groupby([..., ...])
# ToDo: Faça a agregação de soma dos dados agrupados 
# dados_queimadas_soma_dos_anos = dados_queimadas_soma_dos_anos.[ ... ]
# dados_queimadas_soma_dos_anos
# Veja que a tabela acima para entender a diferença com a tabela gerada
dados_queimadas_soma_dos_anos.reset_index(inplace = True)
dados_queimadas_soma_dos_anos
(
    ggplot(dados_queimadas_soma_dos_anos, aes(x = "year", y = "number", color = "factor(state)"))
        + geom_line()
        + facet_wrap("~state")
        + ggtitle("Variação temporal das identificações de fogo em floresta (1998 à 2017)")
        + xlab("Ano")
        + ylab("Quantidade de identificações")
        + theme(axis_text_x = element_text(angle = 45))
)
# ToDo: Encontrar a maior quantidade de identificações já feitas
# dados_queimadas_soma_dos_anos[ ... ]
def ranking_by_year(dataframe, year):
    """Gera um gráfico de barra utilizando a ordem crescente dos dados
    
    Args:
        dataframe (pandas.DataFrame): Conjunto de dados de fogos na floresta
        year (int): ano que se deseja visualizar
    Returns:
        plotnine.ggplot.ggplot: Figura gerada
    """
    
    _df = dataframe[dataframe["year"] == year].sort_values(by = "number")
    _df["state"] = pd.Categorical(_df["state"], categories=_df["state"].unique()[::-1])
    
    return (
        ggplot(_df.sort_values(by = "number"), aes(x = "state", y = "number", color = "state")) 
            + geom_bar(stat = "identity")
            + ylab("Ano")
            + xlab("Quantidade de identificações")
            + ggtitle(f"Ranking das maiores identificações ({year})")
            + coord_flip()
    )
# ToDo: Visualizar o ranking de identificações para os anos de 1999, 2005, 2010 e 2015
# Dica: Consulte a docstring (Texto abaixo da criação da função) para entender seus parâmetros
# ranking_by_year(dados_queimadas_soma_dos_anos, ...)
estados_amazonia_legal = [
    "Acre", "Amapa", "Amazonas", "Mato Grosso", "Pará", "Rondonia", "Roraima", "Tocantins", "Maranhao"
]
# ToDo: Filtre os dados para somente aqueles que estão dentro dos estados da Amazônia Legal
# dados_queimadas_amazonia_legal = dados_queimadas[dados_queimadas[ ... ].isin(estados_amazonia_legal)]
# dados_queimadas_amazonia_legal.head(10)
# ToDo: Aplique a agregação de soma no atributo `number`
# dados_queimadas_amazonia_legal_somageral = dados_queimadas_amazonia_legal["number"].[ ... ]
# dados_queimadas_amazonia_legal_somageral
# ToDo: Filtre os dados em que o atributo `state` NÃO pertenca a Amazônia Legal
# Note o ~ dentro do comando, ele é vital para o funcionamento dessa operação

# dados_queimadas_naoamazonia_legal = dados_queimadas[~dados_queimadas[ ... ].isin( ... )]
# dados_queimadas_naoamazonia_legal.head(10)
# ToDo: Aplique uma agregação de soma
# dados_queimadas_naoamazonia_legalsomageral = dados_queimadas_naoamazonia_legalsomageral[ ... ].[]
# dados_queimadas_naoamazonia_legalsomageral
# ToDo: Crie um pandas.DataFrame com os atributos `local` e `quantidade`, onde:
#  - local: Você vai inserir nessa coluna os locais ("Amazônia legal", "Demais estados")
#  - quantidade: Você vai inserir nessa coluna as quantidade de identificações em cada local (dados_queimadas_amazonia_legal_somageral e dados_queimadas_naoamazonia_legalsomageral)

# diferencas = pd.DataFrame({
#     ...: ...,
#     ...: ...
# })
# diferencas
# ToDo: Crie um gráfico de barra com o dataframe que você gerou anteriormente
# ggplot(diferencas, aes(x = "...", y = "...")) + geom_bar(stat = "identity")