!pip install iso3166
from iso3166 import countries_by_name
import spacy
spacy_model = spacy.load("en_core_web_lg")
from IPython.display import HTML
import pickle
import re
import string
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment import vader

from sklearn.metrics.pairwise import cosine_similarity
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 1000

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

class Paper:
    def __init__(self,
               paper_id,
               doi,
               publish_time,
               journal,
               affiliations,
               source_x,
               title,
               authors,
               abstract,
               text,
               **extras
               ):
        self.paper_id = str(paper_id)
        self.doi = 'https://doi.org/' + str(doi)
        self.publish_time = publish_time
        self.journal = journal
        self.affiliations = affiliations
        self.source_x = source_x
        self.title = title
        self.title_embedding = None
        self.authors = authors.split(',')
        self.abstract = abstract
        self.abstract_embedding = None
        self.text = text
        self.text_embedding = None
class TextProcessor:
    DOMAIN_STOP_WORDS = {"abstract", "al", "article", "arxiv", "author", "biorxiv", "background", "chapter", "conclusions", "copyright", "da", "dei", "del", "dell", "della",
                          "delle", "di", "doi", "edition", "et", "fig", "figure", "funder", "holder", "http", "https",
                        'i', 'ii', 'iii','iv','v', 'vii', 'viii', 'ix', 'x', 'xi', 'xii','xiii', 'xiv' "il", "la", "le", "introduction",
                          "license", "medrxiv", "non", "org", "peer", "peer-reviewed", "permission", "preprint", "publication",
                          "pubmed", "reserved", "reviewed", "rights", "section", "summary", "si", "table", "una"}

    STOP_WORDS = set(stopwords.words("english")) | DOMAIN_STOP_WORDS - set(['non', 'no'])

    @staticmethod
    def clean_text(text, spacy_model):
        if len(text) > spacy_model.max_length:
            text = text[:spacy_model.max_length]
        doc = spacy_model(text)
        raw_sentences = []
        clean_sentences = []
        tokenized_sentences = []
        for sent in list(doc.sents):
            raw_sentences.append(sent.string)
            clean_sentence = ""
            tokens = []
            for token in spacy_model(sent.string.strip()):
                if token.pos_ not in ('PUNCT', 'SYM', 'X') \
                and re.match(r"^\d*[a-z][\-.0-9:_a-z]{1,}$", token.text.lower()):
                    clean_sentence += f"{token.text} "
                    if token.text.lower() not in TextProcessor.STOP_WORDS:
                        tokens.append(token.text.lower())

            clean_sentences.append(clean_sentence.strip())
            tokenized_sentences.append(tokens) 
            for raw, cleaned, tokenized in zip(raw_sentences, clean_sentences, tokenized_sentences):
                if len(tokenized) > 3:
                    yield (raw, cleaned, tokenized)

reseach_papers_file_id = '1ZGRWUIEm9VHfW4JraJ6ECUoNdNX5Q9mW'
download_file_from_google_drive(reseach_papers_file_id, 'reseach_papers.pkl')
with open('reseach_papers.pkl', 'rb') as f:
    research_papers = pickle.load(f) 
type(research_papers[0].text)
research_papers[0].text[0]
def get_df(list_of_paper_obj):
    df = pd.DataFrame(data={'title': [" ".join([title[1] for title in paper.title]) for paper in list_of_paper_obj],
                          'abstract': [" ".join([abstract[1] for abstract in paper.abstract]) for paper in list_of_paper_obj],
                          'text': [" ".join([text[1] for text in paper.text]) for paper in list_of_paper_obj],
                          'affiliations': [paper.affiliations for paper in list_of_paper_obj],
                          'publish_time': [paper.publish_time for paper in list_of_paper_obj]
                          })
    df['publish_time'] = df['publish_time'].map(lambda x: re.search('\d{4}', str(x)).group())
    df['publish_time'] = pd.to_datetime(df['publish_time'])

    return df

def count_relative_number_ethics_articles(x):
    articles_matches = [re.findall('ethical|ethician|ethicist|ethics', record) for record in x['title'] + x['abstract']]
    count_ethics_articles = sum([len(match) != 0 for match in articles_matches])
    return pd.Series([count_ethics_articles / x['title'].count(),
                      count_ethics_articles,
                      x['title'].count()
                      ],
                      index=['relative_count', 'ethics_count', 'total_count']
                    )
df = get_df([paper for paper in research_papers 
             if paper.title != [] and paper.publish_time != '' and paper.abstract != []
             ])
grouped = df.groupby(df.publish_time.dt.year).apply(count_relative_number_ethics_articles)
grouped.index = pd.to_datetime(grouped.index, format='%Y')
def plot_relative_number_ethics_articles(grouped_by_year_df, title):
    grouped_data = [go.Scatter(x=grouped_by_year_df.index,
                            y=grouped_by_year_df['relative_count'],
                            text = [f'ethics count: {row["ethics_count"]}<br>total count: {row["total_count"]}' 
                                    for idx, row in grouped_by_year_df.iterrows()],
                            mode='lines+markers',
                            name='ethics articles count'
                            )]
    grouped_data.append(go.Scatter(x=["2002-01-01", "2019-01-01", "2014-01-01"],
                                 y=[-0.005, -0.005, -0.005],
                                 text=["SARS", "COVID-19", "Ebola"],
                                 mode="text",
                                 name='outbreaks'
                                 ))
    layout = go.Layout(title=title, title_x=0.5, xaxis=dict(title='Date'),
                   yaxis=dict(title='Ratio'))
    shapes=[dict(type="line",
            xref="x",
            yref="paper",
            x0="2002-01-01",
            y0=0,
            x1="2002-01-01",
            y1=1,
            line=dict(width=4, dash='dot', color="LightSeaGreen")
        ),
        dict(type="line",
            xref="x",
            yref="paper",
            x0="2019-01-01",
            y0=0,
            x1="2019-01-01",
            y1=1,
            line=dict(width=4, dash='dot', color="MediumPurple")
        ),
        dict(type="line",
            xref="x",
            yref="paper",
            x0="2014-01-01",
            y0=0,
            x1="2014-01-01",
            y1=1,
            line=dict(width=4, dash='dot', color="RoyalBlue")
        )
        ]
    layout['shapes'] = shapes
    layout['template'] = 'plotly_white'
    fig = go.Figure(data=grouped_data, layout=layout)
    return fig
fig = plot_relative_number_ethics_articles(grouped, title='Ratio of Ethics Articles')
display(HTML('<iframe width="1000" height="500" frameborder="0" scrolling="no" src="https://plotly.com/~kfrid/1.embed"></iframe>'))
def map_tags(string, regex_dict):
    matches = []
    for regex, phrase in regex_dict.items():
        for match in re.findall(regex, string):
            matches.append(phrase)
    return list(set(matches))
ethical_principles = {re.compile(r'\bmax[\w\s]+benefit[s]?\b|\bincreas[\w\s]+benefit[s]?\b|\bbenefit[s\s]+max[\w]*\b|\bbeneficen[\w]+\b', re.I ): 'beneficence',
                      re.compile(r'\bconflict[sing\s]+interest[s]?\b|\bconflict[sing\s]+obligation[\w]*\b', re.I): 'conflicts of interest',
                      re.compile(r'\binclusiv[\w]*\b', re.I): 'inclusiveness',
                      re.compile(r'\binform[ed\s]+consent\b|\binform[ed\s]+choice[s]?\b|\bindividual[\s]+consent[s]?\b', re.I): 'informed consent',
                      re.compile(r'\bpublic[\s]+trust[\w]*\b', re.I): 'public trust',
                      re.compile(r'\breduc[\w\s]+risk[s]?\b|\bmin[\w\s]+risk[s]?\b|\brisk[s\s]+min[\w]+\b', re.I): 'minimizing risk',
                      re.compile(r'\bconfidentiality\b|\bprivacy\b', re.I): 'confidentiality',
                      re.compile(r'\binterest[s]+defense\b', re.I): 'interest defense',
                      re.compile(r'\bsafeguard[\w]*\b', re.I): 'safeguarding',
                      re.compile(r'\bscientific[\s]+objectiv[\w]*\b', re.I): 'scientific objectivity', 
                      re.compile(r'\btransparency\b', re.I): 'transparency',
                      re.compile(r'\bno[n\s-]*maleficence\b|\bno[n\s-]*abandonment\b|\b[no\s-]*harm[\w]\b', re.I): 'nonmaleficence',
                      re.compile(r'\bjustice\b', re.I): 'justice',
                      re.compile(r'\bself[\s-]*care\b|\bself[\s-]*protect[ion]*\b|\bself[\s-]*defens[es]+\b', re.I): 'self-care'
                     }
def count_tags(x):
    data = []
    indexes = []
    for column in x.columns:
        data.append(x[column].sum())
        indexes.append(f'{column}')
    return pd.Series(data=data,
                    index=indexes
                  ).fillna(0)
df['ethical_principles_tags'] = df['text'].apply(lambda x: map_tags("".join(x), ethical_principles))
df = pd.concat([df['publish_time'], df['ethical_principles_tags'].str.join('|').str.get_dummies()], axis=1)
df = df.set_index('publish_time', drop=True)
grouped = df.groupby(df.index.year).apply(count_tags)
grouped.index = pd.to_datetime(grouped.index, format='%Y')
grouped = grouped.reset_index()
df_melt = pd.melt(grouped, id_vars=['publish_time'], value_vars=list(grouped.columns)[1:], var_name=['tag'])
df_melt['year'] = df_melt['publish_time'].dt.year
df_melt[df_melt['year'] == 1990]
def plot_tags_per_year(df, title):
    fig = go.Figure()
    for step in df['year'].unique():
        sample_df_year = df[df['year'] == step].reset_index(drop=True)
        sample_df_year['relative'] = sample_df_year['value'] / sample_df_year['value'].sum()
        sample_df_year['relative'] = sample_df_year['relative'].fillna(0)
        colors = ['lightslategray',] * len(set(sample_df_year['tag']))


        colors[sample_df_year['relative'].idxmax()] = 'crimson'
        fig.add_trace(
            go.Bar(
                visible=False,
                x=sample_df_year['tag'],
                y=sample_df_year['relative'],
                text=sample_df_year['value'],
                textposition='outside',
                marker_color=colors),
                )

# Make 0 trace visible
    fig.data[0].visible = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
          method="restyle",
          label=str(df['year'].unique()[i]),
          args=["visible", [False] * len(fig.data)],
      )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
      active=0,
      currentvalue={"prefix": " ", "xanchor": "right", "font": {
            "color": '#888',
            "size": 32
          }},
      pad={"t": 1},
      steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=title,
        title_x=0.5,
        template='plotly_white'
    )

    fig.update_xaxes(
      showgrid=False,
      ticks="inside",
      tickson="boundaries",
      ticklen=10,
      tickangle=-10
    )

    fig['layout']['yaxis'].update(title='Ratio of Mentions', range=[0, 1], autorange=False)
    return fig
fig = plot_tags_per_year(df_melt, title='Mentions of Ethical Principles')
iplot(fig)
display(HTML('<iframe width="1000" height="550" frameborder="0" scrolling="no" src="https://plotly.com/~kfrid/18.embed?width=1000&height=550"></iframe>'))
health_measure = {re.compile(r'\bhygiene\b', re.I): 'hygiene',
                  re.compile(r'\bfac[\w\s]+mask[s]?\b|\bmask[sfor\s]+fac[es]+\b', re.I): 'face mask',
                  re.compile(r'\bsurfac[es\s]+clean[ing]?|object[s\s]+clean[ing]\b', re.I): 'objects cleaning',
                  re.compile(r'\bultraviolet[\s]+light[\w]*\b', re.I): 'ultraviolet light',
                  re.compile(r'\bmodif[\w\s]+humidity\b|\bincreas[\w\s]+humidity\b', re.I): 'humidity',
                  re.compile(r'\bcontact[s\s]+trac[\w]*\b|\btrac[eingof\s]+contact[s]?\b', re.I): 'contact tracing',
                  re.compile(r'\b[self-]*isolation\b', re.I): 'isolation',
                  re.compile(r'\bquarantin[\w]*\b', re.I): 'quarantine',
                  re.compile(r'\bsocial[\s]+distans[\w]*\b', re.I): 'social distancing',
                  re.compile(r'\bavoid[ing\s]+crowd[\w]*\b', re.I): 'avoiding crowds',
                  re.compile(r'\bschool[s\s]+measure[s]?\b|\bschool[s\s]+clos[\w]*\b', re.I): 'school closure',
                  re.compile(r'\bentr[yies\s]+screen[ing]?\b|\bexit[s\s]+screen[ing]?\b', re.I): 'entry/exit screening',
                  re.compile(r'\btravel[lsing\s]+restrict[\w]*\b|\bvisit[sing\s]?restricti[\w]*\b', re.I): 'travel restrictions',
                  re.compile(r'\bborder[s\s]+clos[\w]*\b', re.I): 'border closure',
}
df = get_df([paper for paper in research_papers 
             if paper.title != [] and paper.publish_time != '' and paper.abstract != []
             ])
df['health_measure_tags'] = df['text'].apply(lambda x: map_tags(x, health_measure))
df = pd.concat([df['publish_time'], df['health_measure_tags'].str.join('|').str.get_dummies()], axis=1)
df = df.set_index('publish_time', drop=True)
grouped = df.groupby(df.index.year).apply(count_tags)
grouped.index = pd.to_datetime(grouped.index, format='%Y')
grouped = grouped.reset_index()
df_melt = pd.melt(grouped, id_vars=['publish_time'], value_vars=list(grouped.columns)[1:], var_name=['tag'])
df_melt['year'] = df_melt['publish_time'].dt.year
fig = plot_tags_per_year(df_melt, title='Mentions of Public Health Measures')
iplot(fig)
display(HTML('<iframe width="1000" height="500" frameborder="0" scrolling="no" src="https://plotly.com/~kfrid/20.embed"></iframe>'))
country_name = {
    re.compile('^(?:(A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY]))$'):'United States of America',
                re.compile(r'\bUnited[\s]+States[\w]*\b|Chapel[\s]Hill|Irvine|Los Angeles|Baltimore|East Lansing|San Francisco|USA', re.I): 'United States of America',
                re.compile(r'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\sHampshire|New\sJersey|New\sMexico|New\sYork|North\sCarolina|North\sDakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\sIsland|South\sCarolina|South\sDakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\sVirginia|Wisconsin|Wyoming', re.I) : 'United States of America',
                re.compile(r'\bUnited[\s]+Kingdom\b|\bGreat[\s]Britain\b|\bLondon\b|Cambridge|Liverpool|England|UK', re.I): 'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND',
                re.compile(r'\b[\w\s]*China\b|\bROC\b|\bWuhan\b|\bHubei\b|\bBeijing\b', re.I): 'China',
                re.compile(r'\b[\w\s]*Korea\b|Seoul|Suwon|Seongnam', re.I): 'KOREA, REPUBLIC OF',
                re.compile(r'[\w\s]*Netherlands|Utrecht|Leiden|Amsterdam|Rotterdam|Bilthoven', re.I): 'Netherlands',
                re.compile(r'\b[\w\s]*Singapore', re.I): 'Singapore',
                re.compile(r'\b[\w\s]*Germany', re.I): 'Germany',
                re.compile(r'Riyadh|[\w\s]*Saudi Arabia', re.I): 'Saudi Arabia',
                re.compile(r'Hong|Shatin|Kowloon|Pokfulam', re.I): 'Hong Kong',
                re.compile(r'España|Spain', re.I): 'Spain',
                re.compile(r'M[ée]+xico', re.I): 'Mexico',
                re.compile(r'Viet[\s]?nam', re.I): 'Viet Nam',
                re.compile(r'Ouro Preto|Bra[sz]il|Belo Horizonte|Porto Alegre', re.I): 'BRAZIL',
                re.compile(r'Paris|France|Lyon', re.I): 'France',
                re.compile(r'Cape Town', re.I): 'South Africa',
                re.compile(r'Brno|Czechia|Czech Republic', re.I): 'Czechia',
                re.compile(r'IRAN', re.I): 'IRAN, ISLAMIC REPUBLIC OF',
                re.compile(r'Russia|Russian Federation', re.I): 'RUSSIAN FEDERATION',
                re.compile(r'Taiwan', re.I): 'TAIWAN, PROVINCE OF CHINA',
                re.compile(r'Tan[sz]ania', re.I): 'TANZANIA, UNITED REPUBLIC OF'}
def map_geo_tags(text, regex_dict):
    matches = re.findall(r', [\w\s]+\)', text)
    clean_matches = set([re.sub('[\d,\)]', "", text).strip(' ') for text in matches])
    clean_matches_2 = []
    for text in clean_matches:
        for regex, phrase in regex_dict.items():
            if re.match(regex, text):
                text = phrase
            clean_matches_2.append(text.upper())
    return list(set(clean_matches_2))
df = get_df([paper for paper in research_papers 
                          if paper.title != [] and paper.publish_time != '' and paper.text != [] and paper.affiliations != '']
                         )
df.shape
df['geo_tags'] = df['affiliations'].apply(lambda x: map_geo_tags(x, country_name))
df = pd.concat([df[['publish_time']], df['geo_tags'].str.join('|').str.get_dummies()], axis=1)
df.shape
df = df.loc[~(df.iloc[:, 1:]==0).all(axis=1)]
df = df.set_index('publish_time', drop=True)
grouped = df.iloc[:, 1:].groupby(df.index.year).apply(count_tags)
grouped.index = pd.to_datetime(grouped.index, format='%Y')
df_paper_per_country = grouped.transpose().cumsum(axis=1)
df_paper_per_country = df_paper_per_country.transpose().reset_index()
df_paper_per_country.columns
df_paper_melt = pd.melt(df_paper_per_country, id_vars=['publish_time'], value_vars=list(df_paper_per_country.columns)[1:], var_name=['country'])
df_paper_melt['iso'] = df_paper_melt['country'].map(lambda x: countries_by_name.get(x, 'Unknown code')[2])
df_paper_melt = df_paper_melt[df_paper_melt['iso'] != 'k'].reset_index(drop=True)
df_paper_melt['year'] = df_paper_melt['publish_time'].dt.year
df_paper_melt
import plotly.express as px
fig = px.choropleth(df_paper_melt,
                    locations="iso",
                    color='value',
                    hover_name="country",
                    animation_frame='year',
                    range_color=[0, 1000],  color_continuous_scale='deep'
                    )
fig.show()
def vectorize(list_of_triplets):
    print("--start--")
    embeddings = embed([" ".join(text) for _, _, text in list_of_triplets], signature="default", as_dict=True)["default"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        x = sess.run(embeddings)
    print('--end--')
    return x
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
tf.test.gpu_device_name()
url = "https://tfhub.dev/google/elmo/3"
embed = hub.Module(url)
covid_sample_papers = [paper for paper in research_papers 
 if len(re.findall('COVID[-\d]*|2019-nCov|SARS[-\s]?cov[-\s]?2',
                   " ".join([title[0] for title in paper.title]) + " ".join([abstract[0] for abstract in paper.abstract]),
                   re.I)
        ) != 0]
for idx in tqdm(range(len(covid_sample_papers)), total=len(covid_sample_papers)):
    covid_sample_papers[idx].text_embedding = vectorize(covid_sample_papers[idx].text)
del research_papers
covid_sample_file_id = '1H_onqws0ccTAn7a2Xe7S4CTcxfImvEBW'
download_file_from_google_drive(covid_sample_file_id, 'covid_sample.pkl')
with open('covid_sample.pkl', 'rb') as f:
    covid_sample_papers = pickle.load(f) 
len(covid_sample_papers)
text_processor = TextProcessor()
sid = vader.SentimentIntensityAnalyzer()


def color_sid(phrase):
    scores = sid.polarity_scores(phrase)
    pos = scores['pos']
    neu = scores['neu']
    neg = scores['neg']
    return 'rgb(' + f'{neg * 255},'  + f'{pos * 255},' + f'{neu * 255})'
def search_similarity(search_string, papers, n_results, threshold=0.7):
    processed_search_string = list(text_processor.clean_text(search_string, spacy_model))
    search_vect = vectorize(processed_search_string)
    output = ""
    sentences = []
    for paper in papers:
        cosine_similarities = pd.Series(cosine_similarity(search_vect,
                                                      paper.text_embedding
                                                      ).flatten())
        if any(cosine_similarities > threshold):
            output += '<p style="font-family:verdana; font-size:110%;"> '
            title = str(paper.title[0][0]) if paper.title != [] else "Empty"
            output += " <b>" + f'{title}' + " </b>" + "</br>"
            for i, j in cosine_similarities.nlargest(int(n_results)).iteritems():
                if j > threshold:
                    output += " <b>"+str(round(j, 3))+" </b>"
                    similar_sentences = [paper.text[i][0]]
                    colored_string = "".join([" ".join(['<span style=' + f'"color:{color_sid(word)}; font-size:100%"' + '>' + f"{word}" + '</span>' for word in sentence.split()]) + '</br>' 
                                  for sentence in similar_sentences])
                    output += colored_string
              # output += " ".join([f'<p style="color: "red">" ' + sentence + ' </p>' 
              #                     for sentence in similar_sentences]) \
              #                     + "</br>" 
              # output += " ".join(similar_sentences) + "</br>"
                    sentences.append(similar_sentences)

            output += "</p></hr>"
        
    output = '<h3>Results:</h3>' + output
    return output, sentences
query = "emotional response and psychological health of workers during an outbreak" 
N = "5" 

output, sim_sentences = search_similarity(query, covid_sample_papers, N, 0.7)

display(HTML(output))

import itertools
import networkx as nx
SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS = {"aux", "auxpass", "neg",  "advmod"}

def get_span_for_compound_noun(noun):
    
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in {"compound", "acomp"} , reversed(list(noun.lefts))
        )
    )

    max_i = noun.i + sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in {"compound", "acomp"}, noun.rights
        )
    )
    return (min_i, max_i)


def get_span_for_verb_auxiliaries(verb):


    min_i = verb.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, reversed(list(verb.lefts))
        )
    )
    max_i = verb.i + sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in AUX_DEPS, verb.rights
        )
    )
    return (min_i, max_i)

def subject_verb_object_triples(doc):
  
    sents = doc.sents

    for sent in sents:
        start_i = sent[0].i
        verbs = get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = get_objects_of_verb(verb)
            if not objs:
                continue

            verb_span = get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i : verb_span[1] - start_i + 1]
            for subj in subjs:
                if subj not in text_processor.STOP_WORDS:
                    subj = sent[
                    get_span_for_compound_noun(subj)[0]
                    - start_i : subj.i
                    - start_i
                    + 1
                ]
            for obj in objs:
                if obj.pos_ == "VERB":
                    continue
                elif obj.pos_ == "NOUN" and obj not in text_processor.STOP_WORDS:
                    span = get_span_for_compound_noun(obj)
            
                    obj = sent[span[0] - start_i : span[1] - start_i + 1]

                    if len(obj) != 0 and len(subj) != 0:
                        yield (subj, verb, obj)




def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]

def get_main_verbs_of_sent(sent):
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos_ == 'VERB' 
        and tok.dep_ not in AUX_DEPS
    ]


def get_subjects_of_verb(verb):
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb):
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp")
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs
def plot_graph(df, node_filter, depth=2, seed=3):
    shells = [[node_filter], list(set(df[['subject', 'object']].values.flatten()) - {node_filter})]
    if depth == -1:
        G = nx.from_pandas_edgelist(df, 'subject', 'object',  create_using=nx.MultiDiGraph())
    else:
        G_full = nx.from_pandas_edgelist(df, 'subject', 'object',  create_using=nx.MultiDiGraph())
        G = nx.ego_graph(G_full, node_filter, radius=depth)
  
    pos = nx.drawing.layout.spring_layout(G, scale=2, k=1.5, seed=seed)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    traceRecode = []  
  

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                          mode='lines',
                          line={'width': 1.5},
                          marker=dict(color='LightGrey'),
                          line_shape='spline',
                          opacity=1
                          )
        traceRecode.append(trace)
        index = index + 1


    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 6, 'color': 'DarkGrey', 'symbol': "circle-dot"},
                                    opacity=1)

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = df[(df['subject'] == edge[0]) & (df['object'] == edge[1])]['relation'].values[0]
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([edge[0] + '---' + hovertext + '---'+ edge[1]])
        index = index + 1

    traceRecode.append(middle_hover_trace)

    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="middle center",
                            hoverinfo="text", marker={ 'size': [], 'color': ['LightBlue'] * len(G.nodes())}, opacity=1)
    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        
        text = node
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple(["<br>".join(text.split())])
        node_trace['marker']['size'] += tuple([max(np.log(len(nx.descendants(G, node))) * 10, 10)])
        index = index + 1

    traceRecode.append(node_trace)

    figure = {
        "data": traceRecode,
        "layout": go.Layout(title=f'{node_filter} connections',
                            title_x=0.5,
                            showlegend=False, hovermode='closest',
                            template = 'plotly_white',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=1000,
                            clickmode='select+event',
                            annotations=[
                                dict(
                                    x=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    y=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                    ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2 ,
                                    ay=(G.nodes[edge[0]]['pos'][1]  + G.nodes[edge[1]]['pos'][1]) / 2, xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=10,
                                    arrowwidth=1,
                                    arrowcolor="LightGrey",
                                    opacity=1,
                                    
                                ) for edge in G.edges]
                            )}

    return figure
svo = []
for paper in tqdm(covid_sample_papers, total=(len(covid_sample_papers))):
    sentence = ". ".join([triplet[1] for triplet in paper.text])
    if len(sentence) > 1000000:
        for i in range(0, len(sentence), 1000000):
            svo += list(subject_verb_object_triples(spacy_model(sentence[i:i+1000000])))
    else:
        svo += list(subject_verb_object_triples(spacy_model(sentence)))
data = [(" ".join([t.text for t in triplet[0]]).strip(),
           " ".join([t.text for t in triplet[1]]).strip(),
           " ".join([t.text for t in triplet[2]]).strip()
           ) for triplet in svo]
df = pd.DataFrame(data=data, columns=['subject', 'relation', 'object'])
df = df.drop_duplicates(keep='first').reset_index(drop=True)
df[df['subject'].str.find('WHO') != -1]
configure_plotly_browser_state()
fig = plot_graph(df, 'WHO', depth=1, seed=8)
display(HTML('<iframe width="1000" height="1000" frameborder="0" scrolling="no" src="https://plotly.com/~kfrid/79.embed"></iframe>'))
display(HTML('<iframe width="1000" height="1000" frameborder="0" scrolling="no" src="https://plotly.com/~kfrid/85.embed"></iframe>'))