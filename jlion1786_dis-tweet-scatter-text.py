%matplotlib inline

import scattertext as st

import re, io

import pandas as pd

from pprint import pprint

from scipy.stats import rankdata, hmean, norm

import spacy

import os, pkgutil, json, urllib

from urllib.request import urlopen

from IPython.display import IFrame

from IPython.core.display import display, HTML

from scattertext import CorpusFromPandas, produce_scattertext_explorer

display(HTML("&lt;style>.container { width:98% !important; }&lt;/style>"))
df = pd.read_json('../input/fin_dis_tweet.json')
#Langauge comparison

scatter_lang = df.loc[:,['language','clean_translated_text']]
nlp = st.WhitespaceNLP.whitespace_nlp_with_sentences

scatter_lang['parsed'] = scatter_lang.clean_translated_text.apply(nlp)
stopwords = ['disneyland','THIS','forum','disney','going','time','resort','tokyo','paris','california','disneysea','florida','disneylands','youre',

                 'mouse','anaheim','disneys','parks','angeles', '𝚃𝙷𝙸𝚂', 'park', 'forwarding', 'fortified', 'just', 'forward','𝙻𝙾𝙾𝙺','forwardlike','forthcoming',

                 'youd', 'world', 'year', 'week', 'day', 'forums', 'forththey', 'forwinnie', '𝙥𝙤𝙨𝙩𝙚𝙙', 'forth', 'forwith', 'fortnights','fortnight','fortnite',

                  'fortitude','fortress','fortunate','fort','fortunately','fortune','forsure','forwar','forwards','𝙘𝙩𝙤𝙗𝙚𝙧','forzen','forsee','foshan','forrest', 

                  'fosho','foster', 'forreal', 'this','with', 'that', 'been', 'have', 'from','when','japan', 'into','your', 'disneyland','THIS','forum','disney','going','time','resort','tokyo','paris','california','disneysea','florida','disneylands','youre',

                 'mouse','anaheim','disneys','parks','angeles', '𝚃𝙷𝙸𝚂', 'park', 'forwarding', 'fortified', 'just', 'forward','𝙻𝙾𝙾𝙺','forwardlike','forthcoming',

                 'youd', 'world', 'year', 'week', 'day', 'mickey', 'tdr', 'tdr_md', 'minnie' ,'forums', 'forththey', 'forwinnie', '𝙥𝙤𝙨𝙩𝙚𝙙', 'forth', 'forwith',

                 'gran', 'graclas', 'gras', 'grafrom', 'tdr_now','wwwwww', 'gotten', 'graba', 'urayasu', 'android', 'atdisneyland', 'from', 'this','with',

               'that','because', 'have', 'chiba', 'there', 'zhao', 'land']
corpus = st.CorpusFromParsedDocuments(scatter_lang, category_col='language', parsed_col='parsed').build().remove_terms(stopwords, ignore_absences=True)
html = st.produce_scattertext_explorer(corpus,

                                       category='en',

                                       category_name='en',

                                       not_category_name='ja',

                                       use_full_doc=True,

                                       minimum_term_frequency=5,

                                       pmi_filter_thresold=10,

                                       term_ranker=st.OncePerDocFrequencyRanker,

                                       width_in_pixels=1000,

                                       sort_by_dist=False)

file_name = 'language_compare.html'

open(file_name, 'wb').write(html.encode('utf-8'))

IFrame(src=file_name, width = 1200, height=700)