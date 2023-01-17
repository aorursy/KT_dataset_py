#hide_input
%load_ext autoreload
%autoreload 2
!pip install TinyDB
#hide
#carregando as dependências
import pandas as pd
import altair as alt
from tinydb import TinyDB, Query
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import display
import json
alt.data_transformers.disable_max_rows()
#hide
#setup inicial de algmas configurações 
default_color_1 = '#323443'
default_color_2 = '#E7997A'
metadados_ficha_catalografica_path = "../input/lexml-brasil-acervo/metadados_ficha_catalografica/metadados_ficha_catalografica.json"
with open(metadados_ficha_catalografica_path, 'r') as f:
    metadados_ficha_catalografica = json.load(f)
with open("./metadados_ficha_catalografica.json", 'w') as f:
    json.dump(metadados_ficha_catalografica, f)
db = TinyDB("./metadados_ficha_catalografica.json") 
#persiste todos os registros do banco num objeto
total_normativos = db.all()
#collapse
def alter_cols(df: pd.DataFrame) -> pd.DataFrame:
    """faz um tratamento nas colunas dataPublished e legislationType"""
    normas = df.copy()
    df['datePublished'] = pd.to_datetime(df['datePublished'])
    df['legislationType'] = df['legislationType'].apply(lambda x : x.split("/")[-1] if x else np.nan)
    return df

def load_into_dataframe(records: list) -> pd.DataFrame:
    """carrega os dados em um dataframe"""
    # https://stackoverflow.com/questions/5352546/extract-subset-of-key-value-pairs-from-python-dictionary-object/5352658
    content = [{k: data.get(k, None) for k in ('legislationIdentifier', 'legislationType', 'description',  'keywords',  'datePublished')} for data in records]
    df = pd.DataFrame(content)
    df = alter_cols(df)
    return df
df = load_into_dataframe(total_normativos)
#hide
df = df.assign(year=df.datePublished.dt.year)
df = df.drop(df[df['legislationType'].isnull()].index)
df.loc[df['legislationType'] == 'Decreto_legislativo', 'legislationType'] = 'Decreto Legislativo'
df.loc[df['legislationType'] == 'Ordinary_law', 'legislationType'] = 'Lei Ordinária'
df.loc[df['legislationType'] == 'Constitution', 'legislationType'] = 'Constituição'
df.loc[df['legislationType'] == 'Medida_provis%C3%B3ria', 'legislationType'] = 'Medida Provisória'
df.loc[df['legislationType'] == 'Lei_complementar', 'legislationType'] = 'Lei Complementar'
df.loc[df['legislationType'] == 'Lei_delegada', 'legislationType'] = 'Lei Delegada'
df.loc[df['legislationType'] == 'Constitutional_amendment', 'legislationType'] = 'Emenda Constitucional'
df.head(2)
count_normas_by_year = df.groupby(['legislationType', 'year'])['legislationIdentifier']\
    .count()\
    .reset_index()\
    .rename(columns={'legislationIdentifier' : 'quantitativo'})
#hide_input
alt.Chart(count_normas_by_year).mark_area(opacity=0.35).encode(
    alt.X("year:O"),
    alt.Y("quantitativo:Q", stack=None),
    alt.Color('legislationType:N', legend=alt.Legend(title="Tipo do Normativo"), scale=alt.Scale(scheme='dark2')),
    tooltip=['legislationType', 'quantitativo']
).interactive()
df.loc[0, 'keywords']
tags = df.explode('keywords').copy()
tags.head(2)
#collapse
def plot_topN_tags(data: pd.DataFrame, field: str, N: int) -> None:
    """Plota um gráfico de barras horizontais com as Top N tags mais frequentes"""
    _ = pd.DataFrame(data)[field].value_counts()[:N]\
        .reset_index()\
        .rename(columns={'index' : 'keywords', 'keywords' : 'quantitativo'})
    chart = alt.Chart(_)\
        .mark_bar()\
        .encode(
            alt.X('quantitativo'),
            alt.Y("keywords", sort='-x'),
            tooltip='quantitativo'
        )\
        .properties(height=700)\
        .configure_mark(color=default_color_2)    
    display(chart)
plot_topN_tags(tags, 'keywords', 25)
#collapse
num_tags = tags['keywords'].nunique()
#collapse
count_clf_tags = pd.DataFrame(tags.groupby('keywords')['legislationIdentifier']\
                              .count()\
                              .sort_values(ascending=False)\
                              .copy())\
                              .rename(columns={'legislationIdentifier' : 'quantitativo'})
count_clf_tags = count_clf_tags.assign(
    cumulative_sum=count_clf_tags.quantitativo.cumsum())
count_clf_tags = count_clf_tags.assign(
    cumulative_perc= 100*count_clf_tags.cumulative_sum/count_clf_tags.quantitativo.sum(),
    rank=range(1, count_clf_tags.shape[0]+1)
)
count_clf_tags['rank'] = count_clf_tags['rank'].astype('category')
count_clf_tags.reset_index(inplace=True)
count_clf_tags.head(25)
#collapse
count_select = count_clf_tags[count_clf_tags['cumulative_perc'] <= 95].copy()
#collapse
chart_all_rank = alt.Chart(count_select).mark_area(
    interpolate='step-after',
    line=True
).encode(
    alt.X('rank:O', axis=alt.Axis(values=[50, 500, 1000, 2000, 3000])),
    alt.Y('quantitativo:Q'),
    color=alt.value(default_color_1),
).properties(
    width=400,
    height=300,
    title='Distribuição da frequência de ocorrência das tags'
).interactive()

chart_top100_rank = alt.Chart(count_select[:100]).mark_area(
    interpolate='step-after',
    line=True
).encode(
    alt.X('rank:O', axis=alt.Axis(values=[10, 20, 30, 50, 75, 100])),
    alt.Y('quantitativo:Q'),
    tooltip='keywords',
    color=alt.value(default_color_2)
).properties(
    width=400,
    height=300,
    title='Distribuição da frequência de ocorrência das TOP 100 tags'
).interactive()
alt.vconcat(chart_all_rank, chart_top100_rank)
#chart_all_rank | chart_top100_rank
#collapse
top50p_tags = count_clf_tags[count_clf_tags['cumulative_perc'] <= 50.00]\
    ['keywords'].values\
    .tolist() #lista das tags amostradas
#faz um recorte nos dados que contenham as tags selecionadas
sampling_tags = tags[tags['keywords'].isin(top50p_tags)].copy() 
#numero de tags selecionadas
num_tags_sampled = sampling_tags['keywords'].nunique()
#quantitativo de normas que serão excluídas da POC
num_diff_normas = (tags['legislationIdentifier'].nunique()-sampling_tags['legislationIdentifier'].nunique()) 
#percentual de normas que serão excluídas da POC
percent_diff_normas = num_diff_normas/tags['legislationIdentifier'].nunique()*100 
#collapse
def most_common_tags_frequency(s: pd.Series) -> pd.DataFrame:
    #faz uma contagem da ocorrência dos quantitativos de tags por normativos
    contador_tags = Counter()
    for _ in s:
        contador_tags[_] += 1
    #cria um dataframe para armazenar as informações estruturadas acima
    plot_count = []
    for _ in contador_tags.most_common(20): #TOP 20 ocorrências mais frequentes
        plot_count.append([_[0], _[1]])
    quantidade_tags_mais_frequentes = pd.DataFrame(plot_count, columns=['n° de tags por norma', 'n° de ocorrências'])
    quantidade_tags_mais_frequentes['n° de tags por norma'] = quantidade_tags_mais_frequentes['n° de tags por norma'].astype('category')
    return quantidade_tags_mais_frequentes

#analise para o conjunto original (populacao)
#identifica o número de tags por norma
count_len_pop = df.loc[df['legislationIdentifier']\
                   .isin(sampling_tags['legislationIdentifier']\
                   .unique()), 'keywords'].apply(lambda x : len(x))
quantidade_tags_mais_frequentes_populacao = most_common_tags_frequency(count_len_pop)

#analise para o conjunto amostral
count_len_sampling = sampling_tags.groupby('legislationIdentifier')['keywords'].count()
quantidade_tags_mais_frequentes_sample = most_common_tags_frequency(count_len_sampling)

#graficos
#hide
top20_pop_chart = alt.Chart(quantidade_tags_mais_frequentes_populacao, title='TOP 20 quantidades de tags por norma na população.')\
        .mark_bar(color=default_color_2)\
        .encode(
            alt.X('n° de ocorrências'),
            alt.Y("n° de tags por norma", sort='-x')
        )
#hide_input
top20_sample_chart = alt.Chart(quantidade_tags_mais_frequentes_sample, title="TOP 20 quantidades de tags por norma na amostragem.")\
        .mark_bar(color=default_color_1)\
        .encode(
            alt.X('n° de ocorrências'),
            alt.Y("n° de tags por norma", sort='-x'))
median_num_tags_norma_original = np.median(count_len_pop)
median_num_tags_norma_sample = np.median(count_len_sampling)
#hide_input
top20_pop_chart | top20_sample_chart