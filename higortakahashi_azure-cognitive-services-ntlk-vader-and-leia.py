# Importando os módulos necessários

import pandas as pd

import datetime

import time

import nltk

from azure.core.credentials import AzureKeyCredential

from azure.ai.textanalytics import TextAnalyticsClient

from leia import SentimentIntensityAnalyzer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pandas_profiling import ProfileReport

from textblob import TextBlob

nltk.download('vader_lexicon')



# Carregando o arquivo .CSV

dfnoticias = pd.read_csv(r'Historico_de_materias.csv')



# Criando um objeto para extrair o perfil dos dados

perfil = ProfileReport(

    dfnoticias,

    title='Perfil dos dados',

    html={

        'style': {

            'full_width': True}})



# Exibindo o perfil dos dados brutos

perfil.to_file(output_file="dadosbrutos.html")



# Remove alguns caracteres especiais nos textos

dfnoticias.replace(to_replace=r'[\n\r\t]', value='', regex=True, inplace=True)



# Remove as linhas duplicadas

dfnoticias.drop_duplicates(subset=['titulo'], keep='first', inplace=True)

dfnoticias.drop_duplicates(

    subset=['conteudo_noticia'],

    keep='first',

    inplace=True)



# Remove as linhas com conteúdo vazio

dfnoticias.replace("", float("NaN"), inplace=True)

dfnoticias.dropna().empty

dfnoticias.drop(dfnoticias[dfnoticias['data'].isnull()].index, inplace=True)

dfnoticias.drop(

    dfnoticias[dfnoticias['conteudo_noticia'].isnull()].index, inplace=True)



# Convertendo o objeto para data

dfnoticias['data_convertida'] = pd.to_datetime(dfnoticias['data'])



# Alterando a máscara de visualização da data para dia/mês/ano

dfnoticias['data_convertida'].apply(lambda x: x.strftime('%d/%m/%Y'))



# Atualiza o índice

dfnoticias.reset_index(drop=True, inplace=True)



# Verificando a quantidade de registros após a limpeza

len(dfnoticias)



# Criando um objeto para extrair o perfil dos dados após a remoção da sujeira

perfil = ProfileReport(

    dfnoticias,

    title='Perfil dos dados',

    html={

        'style': {

            'full_width': True}})



# Gera um arquivo HTML com o perfil dos dados depois da remoção da sujeira

perfil.to_file(output_file="limpo.html")



# Configurando o endereço da API dos serviços cognitivos do Azure

endpoint = ''

key = ''



# Executando a autenticação no Azure

def authenticate_client():

    ta_credential = AzureKeyCredential(key)

    text_analytics_client = TextAnalyticsClient(

        endpoint=endpoint, credential=ta_credential)

    return text_analytics_client



client = authenticate_client()



# Criando uma função que consome a API do Azure e retorna valores

# relacionados à análise de sentimentos

def sentiment_analysis(client, doc):

    documents = [doc]

    response = client.analyze_sentiment(documents=documents)[0]

    return response



# Função para extrair a polaridade do texto consumido pela API do Azure

def sentiment_label(text):

    try:

        return sentiment_analysis(client, text).sentiment

    except BaseException:

        return None



# Rodando a função de classificação do texto

dfnoticias['polaridade_azure_titulo'] = dfnoticias['titulo'].apply(

    sentiment_label)

dfnoticias['polaridade_azure_noticia'] = dfnoticias['conteudo_noticia'].apply(

    sentiment_label)



# Criando as colunas vazias que armazenarão os textos traduzidos

dfnoticias['content'] = ''

dfnoticias['title'] = ''



# Traduzindo a coluna de assuntos

for index, row in (dfnoticias.iterrows()):

    if dfnoticias.at[index, str(

            'title')] == '':  # verifica se a tradução foi feita

        # armazena o texto em português que será traduzido

        translation = TextBlob(dfnoticias.iloc[index]['titulo'])

        # usa a API do Google para fazer a tradução

        en_blob = translation.translate(from_lang='pt', to='en')

        # essa pausa é obrigatória para evitar o bloqueio do IP por excesso de

        # uso

        time.sleep(0.25)

        dfnoticias.at[index, str('title')] = str(

            en_blob)  # grava o texto traduzido



# Traduzindo a coluna de notícias

for index, row in (dfnoticias.iterrows()):

    if dfnoticias.at[index, str(

            'content')] == '':  # verifica se a tradução foi feita

        # armazena o texto em português que será traduzido

        translation = TextBlob(dfnoticias.iloc[index]['conteudo_noticia'])

        # usa a API do Google para fazer a tradução

        en_blob = translation.translate(from_lang='pt', to='en')

        # essa pausa é obrigatória para evitar o bloqueio do IP por excesso de

        # uso

        time.sleep(0.25)

        dfnoticias.at[index, str('content')] = str(

            en_blob)  # grava o texto traduzido



# Função para trazer os valores de polaridade e subjetividade de um texto

# usando TextBlob

def sentiment_calc_textblob(text):

    try:

        return TextBlob(text).sentiment

    except BaseException:

        return None



# Extraindo a pontuação de sentimento das colunas de assunto e notícias (traduzidos

# para o inglês)

dfnoticias['pontuacao_titulo'] = dfnoticias['title'].apply(

    sentiment_calc_textblob)

dfnoticias['pontuacao_noticia'] = dfnoticias['content'].apply(

    sentiment_calc_textblob)



# Colocando os valores de polaridade e subjetividade em suas respectivas

# colunas

dfnoticias[['textblob_titulo', 'subjetividade_textblob_titulo']] = pd.DataFrame(

    dfnoticias.pontuacao_titulo.tolist(), index=dfnoticias.index)

dfnoticias[['textblob_noticia', 'subjetividade_textblob_noticia']] = pd.DataFrame(

    dfnoticias.pontuacao_noticia.tolist(), index=dfnoticias.index)



# Remove as colunas de pontuação, uma vez que já são desnecessárias

dfnoticias.drop(

    columns=[

        'pontuacao_titulo',

        'pontuacao_noticia'],

    inplace=True)



# Cria uma instância da função de análise de sentimento usando NTLK/VADER

sid = SentimentIntensityAnalyzer()



# Função para trazer os valores de polaridade e subjetividade de um texto

# usando NTLK/VADER

def sentiment_calc_vader(text):

    try:

        ss = sid.polarity_scores(text)['compound']

        return ss

    except BaseException:

        return None



# Extraindo a pontuação de sentimento das colunas de assunto e notícias (traduzidos para o

# inglês)

dfnoticias['vader_en_titulo'] = dfnoticias['title'].apply(

    sentiment_calc_vader)

dfnoticias['vader_en_noticia'] = dfnoticias['content'].apply(

    sentiment_calc_vader)



# Carregando o módulo de análise de sentimento da biblioteca LeIA

s = SentimentIntensityAnalyzer()



# Função para trazer os valores de polaridade e subjetividade de um texto

# usando LeIA

def sentiment_calc_leia(text):

    try:

        ss = s.polarity_scores(text)['compound']

        return ss

    except BaseException:

        return None



# Extraindo a pontuação de sentimento das colunas de assunto e notícias em

# português

dfnoticias['vader_pt_titulo'] = dfnoticias['titulo'].apply(

    sentiment_calc_leia)

dfnoticias['vader_pt_noticia'] = dfnoticias['conteudo_noticia'].apply(

    sentiment_calc_leia)



# Selecionando as colunas relevantes antes de exportá-las para um arquivo

# CSV.

dfnoticias = dfnoticias[['data_convertida',

                         'url_noticia',

                         'url_noticia_curto',

                         'assunto',

                         'titulo',

                         'conteudo_noticia',

                         'title',

                         'content',

                         'polaridade_azure_titulo',

                         'polaridade_azure_noticia',

                         'textblob_titulo',

                         'textblob_noticia',

                         'vader_en_titulo',

                         'vader_en_noticia',

                         'vader_pt_titulo',

                         'vader_pt_noticia']]



# Exportando as classificações para um arquivo .CSV.

dfnoticias.to_csv(r'resultado.csv', sep=';',

                  index=False, decimal=',')



# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based

# Model for Sentiment Analysis of Social Media Text. Eighth International

# Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June

# 2014.



# @misc{Almeida2018,

# author = {Almeida, Rafael J. A.},

# title = {LeIA - Léxico para Inferência Adaptada},

# year = {2018},

# publisher = {GitHub},

# journal = {GitHub repository},

# howpublished = {\url{https://github.com/rafjaa/LeIA}}

# }