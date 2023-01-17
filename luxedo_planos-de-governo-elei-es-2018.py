from collections import OrderedDict
import os
import string

import cycler
import gensim
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import PyPDF2
from sklearn import feature_extraction, decomposition
import unidecode
import warnings
import wordcloud

%matplotlib inline
nltk.download("stopwords", "punkt")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
color = plt.cm.tab20b(np.linspace(0, 1, 15))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
warnings.filterwarnings('ignore')
STOPWORDS = unidecode.unidecode(" ".join(nltk.corpus.stopwords.words("portuguese"))).split()
PASTA_PROGRAMAS = "../input/"
df = pd.DataFrame(os.listdir(PASTA_PROGRAMAS), columns=["arquivo"])
df = df[df["arquivo"].str.endswith("pdf")]
df["candidato"] = df["arquivo"].str.replace(".pdf", "").str.split("_").str.join(" ").str.strip()
df["arquivo"] = PASTA_PROGRAMAS+df["arquivo"]
df = df.set_index("candidato").sort_index()
df["paginas"] = df["arquivo"].apply(lambda linha: PyPDF2.PdfFileReader(linha).getNumPages())
df.loc["Marina Silva", "paginas"] *=2 # Layout de livreto
def ler_texto(row):
    with open(row, "r") as fp:
        return fp.read()
    
df["texto"] = df["arquivo"].str.replace(".pdf", ".txt").apply(ler_texto)

df.loc["Ciro Gomes", ["texto"]] = df.loc["Ciro Gomes", ["texto"]]\
    .str.replace("\nn \x07", "\n")\
    .str.replace("\n\x07 |\x0cn|\x07|(\\nn)+|\tn", "")  # WTF Ciro?!?

df["texto"] = df["texto"].str.strip().str.replace("\n+", "\n").str.lower().apply(unidecode.unidecode)
df["caracteres"] = df["texto"].apply(len)
def estatisticas_periodo(row):
    lista_de_períodos = nltk.tokenize.sent_tokenize(row["texto"])
    palavras = pd.Series(lista_de_períodos)\
        .str.replace(f"[{string.punctuation}]", "")\
        .apply(nltk.word_tokenize).apply(len)
    pontuacao = pd.Series(lista_de_períodos)\
        .str.replace(f"[^{string.punctuation}]", "")\
        .apply(nltk.word_tokenize).apply(len)
    return pd.Series({
        "lista_de_periodos": lista_de_períodos,
        "periodos": len(palavras),
        "periodos_por_pagina": len(palavras)/row["paginas"],
        "palavras_por_periodo": palavras.mean(),
        "palavras_por_periodo_std": palavras.std(),
        "palavras_por_periodo_max": palavras.max(),
        "pontuacao_por_periodo": pontuacao.mean(),
        "pontuacao_por_periodo_std": pontuacao.std(),
        "pontuacao_por_periodo_max": pontuacao.max()
    })

df = pd.concat([df, df[["texto", "paginas"]].apply(estatisticas_periodo, axis=1)], axis=1)
def estatísticas_palavras(row):
    lista_de_palavras = row[["texto"]].str.replace("\n", " ")\
        .str.replace(f"[{string.punctuation}]", "")\
        .apply(nltk.word_tokenize)
    pontuacao = row[["texto"]].str.replace("\n", " ")\
        .str.replace(f"[^{string.punctuation}]", "")\
        .apply(nltk.word_tokenize).apply(len).values[0]
    palavras = pd.Series(lista_de_palavras.values[0]).apply(len)
    return pd.Series({
        "lista_de_palavras": lista_de_palavras["texto"],
        "palavras": len(palavras),
        "palavras_unicas": len(set(lista_de_palavras.values[0])),
        "palavras_por_pagina": len(palavras)/row["paginas"],
        "pontuacao": pontuacao,
        "pontuacao_por_pagina": pontuacao/row["paginas"],
        "palavras_por_pontuacao": len(palavras)/pontuacao,
        "tamanho_das_palavras": palavras.mean(),
        "tamanho_das_palavras_std": palavras.std(),
    })

df = pd.concat([df, df[["texto", "paginas"]].apply(estatísticas_palavras, axis=1)], axis=1)
df["paginas"].plot.bar(figsize=(15, 10), rot=60, title="Número de Páginas")
df["caracteres"].plot.bar(figsize=(15, 10), rot=60, title="Número de Caracteres")
df["periodos"].plot.bar(figsize=(15, 10), rot=60, title="Número de Períodos")
df["periodos_por_pagina"].plot.bar(figsize=(15, 10), rot=60, title="Períodos por Página")
df["palavras_por_periodo"].plot.bar(figsize=(15, 10), rot=60, title="Períodos por Página", yerr=df["palavras_por_periodo_std"])
df["pontuacao_por_periodo"].plot.bar(figsize=(15, 10), rot=60, title="Pontuação por Período", yerr=df["pontuacao_por_periodo_std"])
df["palavras"].transpose().plot.bar(figsize=(15, 10), rot=60, title="Número de Palavras")
df["palavras_unicas"].transpose().plot.bar(figsize=(15, 10), rot=60, title="Palavras Únicas")
df["palavras_por_pagina"].plot.bar(figsize=(15, 10), rot=60, title="Palavras por Página")
df["pontuacao"].plot.bar(figsize=(15, 10), rot=60, title="Quantidade de Pontuação")
df["pontuacao_por_pagina"].plot.bar(figsize=(15, 10), rot=60, title="Pontuação por Página")
df["palavras_por_pontuacao"].plot.bar(figsize=(15, 10), rot=60, title="Palavras por Pontuação")
df["tamanho_das_palavras"].plot.bar(figsize=(15, 10), rot=60, title="Tamanho Médio das Palavras", yerr=df["tamanho_das_palavras_std"])
df.drop(columns=["arquivo", "texto", "lista_de_periodos", "lista_de_palavras"])
def calcular_palavras_unicas(row):
    bag = set()
    data = []
    for palavra in row:
        bag.add(palavra)
        data.append(len(bag))
    return data
    
df_palavras_unicas = df["lista_de_palavras"].apply(calcular_palavras_unicas).apply(pd.Series).transpose()
select_candidatos = OrderedDict((
    ("Guilherme Boulos", {"c": color[7], "offset": (400, -500)}),
    ("Fernando Haddad 2t", {"c": color[5], "offset": (400, -500)}),
    ("Geraldo Alckmin", {"c": color[6], "offset": (400, -500)}),
    ("Ciro Gomes", {"c": color[2], "offset": (400, -500)}),
    ("Jair Bolsonaro 2t", {"c": color[10], "offset": (400, -500)}),
    ("Joo Amodo", {"c": color[11], "offset": (400, -500)}),
))
fontsize = "x-large"

fig, ax = plt.subplots(figsize=(12, 8))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks(np.arange(0, 40000+1, 10000))
ax.set_yticks(np.arange(0, 6000+1, 2000))
ax.set_xlabel("palavras", fontsize=fontsize)
ax.set_ylabel("palavras sem repetição", fontsize=fontsize)
with pd.plotting.plot_params.use('x_compat', True):
    for candidato, item in select_candidatos.items():
        sr_tmp = df_palavras_unicas[candidato]
        sr_tmp[:40000].plot(c=item["c"], linewidth=20, alpha=0.7, ax=ax)
        y = sr_tmp.max().astype(int)
        x = sr_tmp[sr_tmp == sr_tmp.max()].index[-1]
        candidato_offset = np.array((x, y))+item["offset"]
        if candidato != "Guilherme Boulos":
            ax.annotate(candidato, candidato_offset, fontsize="large")
        else:
            ax.annotate("", (40000, 6100), (37500, 5800), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=5))
            ax.annotate("Guilherme Boulos\nAd Infinitum", (35000, 5400), fontsize="large")
ax.set_title("Palavras únicas ao longo do texto\nPlanos de governo - Eleições 2018", fontsize=fontsize)
df["quebras_de_linha"] = df["texto"].str.count("[\n]")
pontuação = {
    "ponto_final": "[.]",
    "vírgula": "[,]",
    "parenteses": "[()]",
    "ponto_virgula": "[;]",
    "dois_pontos": "[:]",
    "asterisco": "[*]",
    "hifen/menos": "[-]",
    "porcentagem": "[%]",
    "barra": "[\\/]",
    "aspas": "[\"\']",
    "moeda": "[$]",
    "mais": "[+]",
    "exclamação": "[!]",
    "interrogação": "[?]",
    "subtraço": "[_]",
    "colchetes": "[\[\]]",
    "chaves": "[\{\}]",
    "e_comercial": "[&]",
    "cerquilha": "#"
}
for key, value in pontuação.items():
    df[key] = df["texto"].str.count(value)
df[list(pontuação)]\
    .divide(df[list(pontuação)].sum(axis=1), axis=0)\
    [list(pontuação)[:len(pontuação)//2]]\
    .transpose().plot.bar(figsize=(18, 10), title="Pontuações Normalizadas #1", rot=20)
df[list(pontuação)]\
    .divide(df[list(pontuação)].sum(axis=1), axis=0)\
    [list(pontuação)[len(pontuação)//2:]]\
    .transpose().plot.bar(figsize=(18, 10), title="Pontuações Normalizadas #2", rot=20)
select_candidatos = OrderedDict((
    ("Cabo Daciolo", {"c": color[1], "offset": (400, -500)}),
    ("Ciro Gomes", {"c": color[2], "offset": (400, -500)}),
    ("Fernando Haddad 2t", {"c": color[5], "offset": (400, -500)}),
    ("Geraldo Alckmin", {"c": color[6], "offset": (400, -500)}),
    ("Jair Bolsonaro 2t", {"c": color[10], "offset": (400, -500)}),
    ("Vera Lucia", {"c": color[14], "offset": (400, -500)}),
))
fontsize = "large"


fig, ax = plt.subplots(figsize=(12, 8))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks(np.arange(0, len(select_candidatos)+1, 1))
ax.set_xticklabels(list(select_candidatos.keys()), rotation=20, fontsize=fontsize)
ax.set_yticks(np.arange(0, 0.061, 0.02))
ax.tick_params(axis='x', which='both',length=0)


with pd.plotting.plot_params.use('x_compat', True):
    for idx, (candidato, item) in enumerate(select_candidatos.items()):
        sr_tmp = df.loc[candidato]
        ax.bar(idx, sr_tmp["exclamação"]/sr_tmp[list(pontuação)].sum(),color=item["c"])
ax.set_title("Quantidade de exclamações relativas ao total de pontuações\nPlanos de governo - Eleições 2018", fontsize=fontsize)
df[["quebras_de_linha"]+list(pontuação.keys())]
df["lista_de_palavras_stopwords"] = df["lista_de_palavras"].apply(lambda row: [palavra for palavra in row if palavra not in STOPWORDS])
df["palavras_frequentes"] = df["lista_de_palavras_stopwords"].apply(lambda row: [item for item in nltk.FreqDist(row).items() if item[1] >= 4])
df["wordcloud_palavras_frequentes"] = ""
display(HTML("<h3>Palavras mais frequentes</h3>"))
for key, value in df["palavras_frequentes"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    df_candidato = pd.DataFrame(value, columns=["Palavra", "Quantidade"])
    df_candidato["%"] = df_candidato["Quantidade"]/df.loc[key, "palavras"]  # Normalizada pela quantidade de palavras do plano
        
    frequencias = df_candidato[["Palavra", "%"]].set_index("Palavra")["%"].to_dict()
    df.loc[key, "wordcloud_palavras_frequentes"] = wordcloud.WordCloud(
        width=800, height=400, background_color='white', random_state=50)\
        .generate_from_frequencies(frequencias).to_image()
    display(df_candidato.sort_values("Quantidade", ascending=False).head(20))
display(HTML("<h3>Nuvens de palavras mais frequentes</h3>"))
for key, value in df["palavras_frequentes"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    display(df.loc[key, "wordcloud_palavras_frequentes"])
    display(HTML("<hr/>"))
def heatmap(df, title, xlabel, ylabel):
    fig, ax1 = plt.subplots(figsize=(15, 15))
    cax = ax1.imshow(df, interpolation="nearest")
    ax1.grid(True)
    plt.title(title)
    ax1.tick_params("x", labelrotation=-40)
    ax1.set_xticks(np.arange(len(df.columns)))
    ax1.set_yticks(np.arange(len(df.index)))
    ax1.set_xticklabels(df.columns, fontsize=10, horizontalalignment="left")
    ax1.set_yticklabels(df.index,fontsize=10)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.show()


def normalizar_score(row, limite=20):
    sr = pd.Series(dict(row)).sort_values(ascending=False)[:limite]
    return sr/sr.sum()


TOP_PALAVRAS = 25
df_palavras_frequentes = df["palavras_frequentes"].apply(normalizar_score, args=(TOP_PALAVRAS,)).fillna(0)
top_palavras_frequentes = df_palavras_frequentes.sum().sort_values(ascending=False)[:TOP_PALAVRAS].index
top_palavras_frequentes_index = df_palavras_frequentes[top_palavras_frequentes].sum(axis=1).sort_values(ascending=False).index
heatmap(
    df_palavras_frequentes[top_palavras_frequentes].reindex(top_palavras_frequentes_index), 
    f"{TOP_PALAVRAS} Palavras mais frequentes por candidato",
    "Palavra",
    "Candidato")
df_palavras_frequentes
df["bigramas_frequentes"] = df["lista_de_palavras_stopwords"]\
    .apply(lambda row: [item for item in nltk.FreqDist(nltk.bigrams(row)).items() if item[1] >= 4])\
    .apply(lambda row: list(map(lambda x: (" ".join(x[0]), x[1]), row)))
df["wordcloud_bigramas_frequentes"] = ""
display(HTML("<h3>Bigramas mais frequentes</h3>"))
for key, value in df["bigramas_frequentes"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    df_candidato = pd.DataFrame(value, columns=["Bigrama", "Quantidade"]).sort_values("Quantidade", ascending=False)
    df_candidato["%"] = 2*df_candidato["Quantidade"]/df.loc[key, "palavras"]  # Normalizada pela quantidade de palavras do plano
        
    frequencias = df_candidato[["Bigrama", "%"]].set_index("Bigrama")["%"].to_dict()
    df.loc[key, "wordcloud_bigramas_frequentes"] = wordcloud.WordCloud(
        width=800, height=400, background_color='white', random_state=50)\
        .generate_from_frequencies(frequencias).to_image()
    display(df_candidato.head(20))
display(HTML("<h3>Nuvens de bigramas mais frequentes</h3>"))
for key, value in df["bigramas_frequentes"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    display(df.loc[key, "wordcloud_bigramas_frequentes"])
    display(HTML("<hr/>"))
df_bigramas_frequentes = df["bigramas_frequentes"].apply(normalizar_score, args=(TOP_PALAVRAS,)).fillna(0)
top_bigramas_frequentes = df_bigramas_frequentes.sum().sort_values(ascending=False)[:TOP_PALAVRAS].index
top_bigramas_frequentes_index = df_bigramas_frequentes[top_bigramas_frequentes].sum(axis=1).sort_values(ascending=False).index
heatmap(
    df_bigramas_frequentes[top_bigramas_frequentes].reindex(top_bigramas_frequentes_index), 
    f"{TOP_PALAVRAS} Bigramas mais frequentes por candidato",
    "Bigrama",
    "Candidato")
def calcular_pmi(row):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(row)
    finder.apply_freq_filter(4)
    scored = finder.score_ngrams(bigram_measures.pmi)
    return list(map(lambda x: (" ".join(x[0]), x[1]), scored))

df["bigramas_pmi"] = df["lista_de_palavras"].apply(calcular_pmi)
df["wordcloud_bigramas_pmi"] = ""
display(HTML("<h3>Bigramas PMI</h3>"))
for key, value in df["bigramas_pmi"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    df_candidato = pd.DataFrame(value, columns=["Bigrama", "Score"])
    
    frequencias = df_candidato[["Bigrama", "Score"]].set_index("Bigrama")["Score"].to_dict()
    df.loc[key, "wordcloud_bigramas_pmi"] = wordcloud.WordCloud(
        width=800, height=400, background_color='white', random_state=50)\
        .generate_from_frequencies(frequencias).to_image()
    display(df_candidato.head(20))
display(HTML("<h3>Nuvens de bigramas PMI</h3>"))
for key, value in df["bigramas_frequentes"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    display(df.loc[key, "wordcloud_bigramas_pmi"])
    display(HTML("<hr/>"))
df_bigramas_pmi = df["bigramas_pmi"].apply(normalizar_score, args=(TOP_PALAVRAS,)).fillna(0)
top_bigramas_pmi = df_bigramas_pmi.sum().sort_values(ascending=False)[:TOP_PALAVRAS].index
top_bigramas_pmi_index = df_bigramas_pmi[top_bigramas_pmi].sum(axis=1).sort_values(ascending=False).index
heatmap(
    df_bigramas_pmi[top_bigramas_pmi].reindex(top_bigramas_pmi_index), 
    f"{TOP_PALAVRAS} Bigramas PMI por candidato",
    "Bigrama",
    "Candidato")
vetorizador_tfidf = feature_extraction.text.TfidfVectorizer(
    stop_words=STOPWORDS, ngram_range=(1, 3), max_features=df["palavras_unicas"].sum()//len(df))
features = vetorizador_tfidf.fit_transform(df["texto"])
feature_names = vetorizador_tfidf.get_feature_names()
corpus_index = df.index
df_tfidf = pd.DataFrame(features.T.todense(), index=feature_names, columns=corpus_index)
df["termos_tfidf"] = ""
for key in corpus_index:
    df.loc[key, "termos_tfidf"] = df_tfidf[[key]].apply(lambda row: list(item for item in zip(row.index, row.values) if item[1] > 0)).values
df["wordcloud_tfidf"] = ""
display(HTML("<h3>Palavras relevantes TF-IDF</h3>"))
for key, value in df["termos_tfidf"].iteritems():
    df_candidato = pd.DataFrame(value, columns=["Palavra", "Score"])
    frequencias = df_candidato[["Palavra", "Score"]].set_index("Palavra")["Score"].to_dict()
    df.loc[key, "wordcloud_tfidf"] = wordcloud.WordCloud(
        width=800, height=400, background_color='white', random_state=50)\
        .generate_from_frequencies(frequencias).to_image()
    display(HTML(f"<h4>{key}</h4>"))
    display(df_candidato.sort_values("Score", ascending=False).head(20).reset_index(drop=True))
display(HTML("<h3>Nuvens de termos TF-IDF</h3>"))
for key, value in df["termos_tfidf"].iteritems():
    display(HTML(f"<h4>{key}</h4>"))
    display(df.loc[key, "wordcloud_tfidf"])
    display(HTML("<hr/>"))
def heatmap(df, title, xlabel, ylabel):
    fig, ax1 = plt.subplots(figsize=(15, 15))
    cax = ax1.imshow(df, interpolation="nearest")
    ax1.grid(True)
    ax1.tick_params("x", labelrotation=-40)
    ax1.set_xticks(np.arange(len(df.columns)))
    ax1.set_yticks(np.arange(len(df.index)))
    ax1.set_xticklabels(df.columns, fontsize=12, horizontalalignment="left")
    ax1.set_yticklabels(df.index,fontsize=12)
    plt.show()
    

top_termos_tfidf = df_tfidf.sum(axis=1).sort_values(ascending=False)[:TOP_PALAVRAS].index
top_termos_tfidf_index = df_tfidf.transpose()[top_palavras_frequentes].sum(axis=1).sort_values(ascending=False).index

heatmap(
    df_tfidf.transpose()[top_palavras_frequentes].reindex(top_palavras_frequentes_index).drop(index=["Fernando Haddad 1t", "Jair Bolsonaro 1t"]), 
    f"{TOP_PALAVRAS} Termos TF-IDF por candidato",
    "Palavra",
    "Candidato")
modelo_pca = decomposition.PCA(n_components=2)
projecao_pca = modelo_pca.fit_transform(df_tfidf.transpose())
offset = {
    "Vera Lucia": (-0.14, -0.07),
    "Alvaro Dias": (-0.14, -0.15),
    "Jair Bolsonaro 1ºt": (-0.4, 0),
    "Jair Bolsonaro 2ºt": (-0.2, -0.06),
    "Henrique Meirelles": (-0.2, -0.07),
    "João Amoêdo": (-0.16, -0.037),
    "Cabo Daciolo": (0.0, -0.11),
    "João Goulart": (-0.22, -0.07),
    "Eymael": (-0.01, -0.11),
    "Marina Silva": (-0.01, -0.11),
    "Geraldo Alckmin": (-0.16, -0.16),
    "Fernando Haddad 2ºt": (-0.12, -0.06),
    "Guilherme Boulos": (-0.14, -0.06),
}
fig, ax = plt.subplots(figsize=(10, 10))
for idx, (candidato, value) in enumerate(zip(corpus_index, projecao_pca)):
    ax.scatter(value[0], value[1], label=candidato, s=400, linewidth=0.1)
    offset_candidato = np.array((value[0]+0.05, value[1]+0.1))
    offset_candidato += offset.get(candidato, (0, 0))
    if candidato not in ["Jair Bolsonaro 1ºt", "Fernando Haddad 1ºt", "Ciro Gomes"]:
        if candidato == "Marina Silva":
            candidato = "Ciro Gomes/Marina Silva"
        ax.annotate(candidato, offset_candidato,fontsize=13)
ax.set_title("PCA das características TF-IDF dos planos de governo\nEleições 2018")
ax.axis('off')
plt.show()
pd.DataFrame(projecao_pca, columns=("x", "y"), index=corpus_index)
def _substituir_bigramas(row, bigramas):
    b_iter = iter(zip(row, row[1:]))
    for p1, p2 in b_iter:
        bigrama = f"{p1} {p2}"
        if bigrama in bigramas:
            yield bigrama
            try:
                next(b_iter)
            except(StopIteration):
                pass
        else:
            yield p1

            
def substituir_bigramas(row, bigramas):
    return list(_substituir_bigramas(row, bigramas))
    
bigramas = np.unique(np.concatenate(df["bigramas_frequentes"].apply(lambda row: list(map(lambda x: x[0], row))).values))
df["tokens"] = df["lista_de_palavras_stopwords"].apply(substituir_bigramas, args=(bigramas,))
NUM_TOPICS = 4
dictionary = gensim.corpora.Dictionary(df["tokens"])
corpus = [dictionary.doc2bow(text) for text in df["tokens"]]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, random_state=42)
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)