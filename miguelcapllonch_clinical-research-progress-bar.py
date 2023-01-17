# TODO: write your imports here

import os

import json



import pandas as pd



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from nltk.stem import WordNetLemmatizer



from gensim.summarization import summarize

from sklearn.feature_extraction.text import TfidfVectorizer



import pickle as pk

import numpy as np



import warnings

warnings.filterwarnings('ignore')



# path to data

data_dir = '../input/CORD-19-research-challenge'  

keyword_dir = '../input/solidpancovid/solid-pancovid-19_kaggle/keywords'
# As kaggle only allows notebook submissions, all functions should be in the notebook. Just copy your functions and paste them here.

          

def load_data(data_dir):

    """Load data from dataset data directory."""

    sha = []

    full_text = []



    subdir = [x for x in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir,x)) and "embeddings" not in x)]



    print(f"[INFO] Loading data from {data_dir}...")

    # loop through folders with json files

    for folder in subdir:

#             path = os.path.join(data_dir,folder, folder)

        path = os.path.join(data_dir,folder, folder, 'pdf_json')

#        path = os.path.join(data_dir,folder, folder)

        # loop through json files and scrape data

        for file in os.listdir(path):

            file_path = os.path.join(path, file)



            # open file only if it is a file

            if os.path.isfile(file_path):

                with open(file_path) as f:

                    data_json = json.load(f)

                    sha.append(data_json['paper_id'])



                    # combine abstract texts / process

                    combined_str = ''

                    for text in data_json['body_text']:

                        combined_str += text['text'].lower()

                        

                    full_text.append(combined_str)



            else:

                print('[WARNING]', file_path, 'not a file. Check pointed path directory in load_data().')



    loaded_samples = len(sha)

    print(f"[INFO] Data loaded into dataset instance. {loaded_samples} samples added.")

    

    df = pd.DataFrame()

    df['sha'] = sha

    df['full_text'] = full_text

    

    return df



def clean_time(val):

    try:

        return datetime.strptime(val, '%Y-%m-%d')

    except:

        try:

            return datetime.strptime(val, '%Y %b %d')

        except:

            try:

                return datetime.strptime(val, '%Y %b')

            except:

                try:

                    return datetime.strptime(val, '%Y')

                except:

                    try:

                        return datetime.strptime('-'.join(val.split(' ')[:3]), '%Y-%b-%d')

                    except Exception as e:

                        return None
def tokenize_check(text):

    if isinstance(text, str):

        word_tokens = word_tokenize(text)

    elif isinstance(text, list):

        word_tokens = text

    else:

        raise TypeError

    return word_tokens

    



def remove_stopwords(text, remove_symbols=False):

    """ Tokenize and/or remove stopwords and/or unwanted symbols from string"""

    list_stopwords = set(stopwords.words('english'))

    # list of signs to be removed if parameter remove_symbols set to True

    list_symbols = ['.', ',', '(', ')', '[', ']']

    

    # check input type and tokenize if not already

    word_tokens = tokenize_check(text)



    # filter out stopwords

    text_without_stopwords = [w for w in word_tokens if not w in list_stopwords] 

    

    if remove_symbols is True:

        text_without_stopwords = [w for w in text_without_stopwords if not w in list_symbols]

    

    return text_without_stopwords



# from nltk.stem import WordNetLemmatizer 



def lemmatize(text):

    """ Tokenize and/or lemmatize string """

    lemmatizer = WordNetLemmatizer()

    

    # check input type and tokenize if not already

    word_tokens = tokenize_check(text)

    

    lemmatized_text = [lemmatizer.lemmatize(w) for w in word_tokens]

    

    return lemmatized_text



def flatten_list(l):

    """ Flatten a list of lists """

    return [item for sublist in l for item in sublist]



def dfkw_cleaning(df):

    """ Clean df for a better keyword finding """



    # Data cleaning:

    # Turn df into a dictionary with a list of key phrases

    # Lower all of them and remove null values

    dfd = {k: [x.lower() for x in v if not pd.isnull(x)] for k, v in df.to_dict('list').items()}

    

    for k, v in dfd.items():



        # Split terms that are in brackets, like "Acyclovir (Aciclovir)"

        v = flatten_list([x.replace('\xa0', '').replace(')', '').split('(') for x in v]) 

        # Remove redundant values (i.e., ['coronavirus', 'coronavirus disease'] can be left as ['coronavirus']; the element 'coronavirus disease' is useless)

        v = [x for x in v if not any([y in x for y in [z for z in v if z != x]])]



        # Store the updated v

        dfd[k] = v



    # Return the clean df

    return pd.DataFrame.from_dict({k: pd.Series(v) for k, v in dfd.items()})



def find_keywords(text, df):

    """ Find relevant papers for the categories in df

    Returns a dictionary with the paper id's that match the categories

    It also stores the sentences where the matches have been found. This can be returned too if so the team decides """



    # Turn df into a dictionary with a list of key phrases

    # Lower all of them and remove null values

    dfd = {k: [x.lower() for x in v if not pd.isnull(x)] for k, v in df.to_dict('list').items()}



    matches = {}

    scores = {}

    

    for k, v in dfd.items():



        # Find matches

        

        for sentence in sent_tokenize(text):

            # Lower-case the sentence for better pattern finding

            sentence_l = sentence.lower()

            # Words have to be tokenized because there are cases like where "sars-cov" is counted where the actual word is "sars-cov-23"

            words = tokenize_check(sentence_l)

            # The condition for a match will be that the word(s) or is (are) in both the tokenized and non-tokenized sentence



            for keyphrase in v:



                # Check that the individual words that compose the key phrase are all 

                # in the words list

                words_in = all([words.count(x) > 0 for x in keyphrase.split(' ')])



                # Check if the keyphrase is in the non-tokenized sentence

                insentence = keyphrase in sentence_l



                # The key phrase is in the sentence if both conditions meet

                insentence = insentence and words_in



                # Now add the match

                if insentence:

                    try:

                        already_a_match = sentence in matches[k]

                    except KeyError:

                        matches[k] = [sentence]

                    else:

                        if not already_a_match:

                            matches[k].append(sentence)

                          

        # score is scaled by the number of values to choose from

        if k in matches:

            scores[k] = len(matches) / len(v)



    # return the keys with the highest score. also return the sentences for this.

    if len(scores.keys()) > 0:

        max_score = list(scores.keys())[np.argmax(scores.values())]

        return max_score, matches[max_score]

    else:

        # Returning np.nan allows detecting these nan's with .isnull()

        return np.nan, np.nan



def kw_match_tables(df):

    """ Build table with boolean values indicating kw matches """



    keywords = {

        'virus': virus_keywords.columns.tolist(), 

        'stage': clinical_stage_keywords.columns.tolist(), 

        'drug': drug_keywords.columns.tolist(), 

    }



    # headers = [item for sublist in [x.columns.tolist() for x in [virus_keywords, clinical_stage_keywords, drug_keywords]] for item in sublist]

    headers = flatten_list([x.columns.tolist() for x in [virus_keywords, clinical_stage_keywords, drug_keywords]])



    # Use titles instead of sha hashes because there's a lot of papers without sha that contain the keywords; yet all papers have a title

    table = pd.DataFrame(False, index=df.title, columns=headers)



    # Fill with True values

    for k, kws in keywords.items():

        for kw in kws:

            table.loc[df[df[k] == kw].title.tolist(), kw] = True



    # Merge

    df = pd.merge(df, table, on='title')



    return df





def summarizeText(text):

    """ 

    Apply gensim summarization function



    "word_count" (=length of summarized text) can be modified as needed 

    """

    try:

        return summarize(text, word_count=200)

    except:

        return text



def tfidfKeywords(df):

    """

    Extract keywords with tfidf



    Step 1: Build corpus from all available fullTexts

    Step 2: Extract keywords iteratively for each paper



    Parameters: Number of keywords to be extracted

    Output: No return values - keywords are written directly into the dataframe

    """



    # PARAMETERS

    numKeywords = 8 # Number of keywords to be extracted



    # Append column to pandas dataframe for storing the keywords

    df["tfidfKeywords"] = "No tf-idf keywords available"





    ### Create corpus from all available fullTexts

    corpus = []

    # Loop through all papers

    for idx in range(0, len(df)):

        # Skip papers without fullText

        if df.iloc[idx]["full_text"] != "":

            # Append fulltext to corpus

            corpus.append(df.iloc[idx]["full_text"])



    import numpy as np

    from sklearn.feature_extraction.text import TfidfVectorizer



    # Initialize tfidf transformer

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85, max_features=20000)



    # Apply tfidf to corpus

    X = vectorizer.fit_transform(corpus)





    ### Extract keywords for each paper

    paperCounter = 0  # count only papers with available fulltext, necessary for row-index of tf-idf matrix

    for i in range(0, len(df)):

        tfidfKeywords = ""

        if df.iloc[i]["full_text"] != "":

            # Extract row from tfidf matrix

            arr = np.squeeze(X[paperCounter].toarray())

            paperCounter += 1



            # Sort entried (descending)

            maxArgs = np.argsort(-arr)



            # Output Keywords

            for j in maxArgs[:numKeywords]:

                tfidfKeywords = tfidfKeywords + " " + vectorizer.get_feature_names()[j]

                #print("Keyword: ", vectorizer.get_feature_names()[j], "   Score: ", arr[j])



            # Write keywords to dataframe

            df.iloc[i, df.columns.get_loc('tfidfKeywords')] = tfidfKeywords

    

    return df



#def visualize_data(data,keywords,summaries):

#    #TODO @Levi @Kwan: visualize data
# keywords that define the virus the paper is about (likely in title)

virus_keywords = pd.read_csv(keyword_dir+'/virus_keywords.csv')



# keywords describing clinical phase

clinical_stage_keywords = pd.read_csv(keyword_dir+'/phase_keywords.csv')



# keywords describing treatment types

drug_keywords = pd.read_csv(keyword_dir+'/drug_keywords.csv')
# try the preloaded dataframe to speed up the process

try:

    df = pk.load(open('df.pkl','rb'))

except:

    # create dataset object

    meta_data = pd.read_csv(data_dir+'/metadata.csv')

    meta_data['publish_time'] = meta_data['publish_time'].apply(clean_time)

    full_texts = load_data(data_dir)



    # merge full text and metadata, so the paper selection can be performed either on full text

    # or abstract, if the full text is not available.

    df = pd.merge(meta_data,full_texts,on='sha',how='outer')

    df['full_text'][df['full_text'].isna()] = df['abstract'][df['full_text'].isna()]



    # drop papers with no abstract and no full text

    df = df.dropna(subset=['abstract','full_text'])

    df = df[df['full_text'] != 'Unknown']

    pk.dump(df,open('df.pkl','wb'))
df.head()
try:

    df = pk.load(open('../input/pickle/df_kw.pkl','rb'))

except:

    # First clean the keyword dataframes for a better keyword finding

    virus_keywords = dfkw_cleaning(virus_keywords)

    clinical_stage_keywords = dfkw_cleaning(clinical_stage_keywords)

    drug_keywords = dfkw_cleaning(drug_keywords)

    # function on full text --> think about applying on full text or on abstract

    df['virus'], df['virus_sentence'] = zip(*df['abstract'].apply(find_keywords, df=virus_keywords))

    df['stage'], df['stage_sentence'] = zip(*df['abstract'].apply(find_keywords, df=clinical_stage_keywords))

    df['drug'], df['drug_sentence'] = zip(*df['abstract'].apply(find_keywords, df=drug_keywords))    

    

    pk.dump(df,open('df_kw.pkl','wb'))
df
try:

    df = pk.load(open('../input/pickle/df_kw_sum.pkl','rb'))

except:

    df['summary'] = df['full_text'].apply(summarizeText)

    pk.dump(df,open('df_kw_sum.pkl','wb'))
if False:

    try:

        df = pk.load(open('../input/pickle/df_kw_sum_kw.pkl','rb'))

    except:

        df = tfidfKeywords(df)

        pk.dump(df,open('df_kw_sum_kw.pkl','wb'))
df
#add some columns to the data

df['publish_time_month'] = df.publish_time.apply(lambda x: x.strftime('%Y-%m') if not pd.isna(x) else np.nan)

df = df.set_index('title')
# !pip install jupyter-plotly-dash
# !conda install -y -c conda-forge jupyter-server-proxy
# !jupyter serverextension install jupyter_server_proxy
!jupyter serverextension enable jupyter_server_proxy
from jupyter_plotly_dash import JupyterDash



import dash

import dash_core_components as dcc

import dash_html_components as html

from dash.dependencies import Input, Output

import pandas as pd

import plotly.express as px

import numpy as np

from datetime import datetime

import traceback



'''

This is a prototype webapp for the CORD19 challenge on Kaggle

This involves a demo pandas dataframe, and sample visualisations

All data here is fictional!

'''



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def make_bubbleplot(df):

    return px.scatter(\

                      pd.pivot_table(df.reset_index(), values=['title'], index=['virus', 'stage', 'publish_time_month'], aggfunc=np.count_nonzero).reset_index().rename(columns={'title':'count'}),\

                      x="publish_time_month", y="stage", color="virus", size='count',

                 hover_name="virus", title='Occurance of research tag per month per phase sized by #Occurances')



app = JupyterDash(__name__)



app.layout = html.Div(children=[

    html.H1(children='COVID-19: Visual Research Exploration Tool'),

    html.Marquee('The data in this tool is fictional!', draggable='true'),

    dcc.Tabs([

        dcc.Tab(label='Overview', children=[    

            dcc.Graph(

            id='stage-plot',

            figure=make_bubbleplot(df)

    )]),

        dcc.Tab(label='Discover', children=[

            html.Div('virus filter'),

            dcc.Dropdown(

                id=f'dropdown-virus',

                options=[{'label': k, 'value':k} for k in df.virus.unique() if not pd.isna(k)],

                multi=True,

                value=[k for k in df.virus.unique()]

            ),

            html.Div('stage filter'),

            dcc.Dropdown(

                id=f'dropdown-stage',

                options=[{'label': k, 'value':k} for k in df.stage.unique()],

                multi=True,

                value=[k for k in df.stage.unique()]

            ),

            html.Div('drug filter'),

            dcc.Dropdown(

                id=f'dropdown-drug',

                options=[{'label': k, 'value':k} for k in df.drug.unique()],

                multi=True,

                value=[k for k in df.drug.unique()]

            ),

            html.Div('x-axis'),

            dcc.Dropdown(

                id='x-axis',

                options=[{'label': k, 'value':k} for k in ['stage', 'virus', 'drug']],

                value='stage'

            ),

            html.Div('hue (color)'),

            dcc.Dropdown(

                id='hue-axis',

                options=[{'label': k, 'value':k} for k in ['stage', 'virus', 'drug']],

                value='virus'

            ),

            # ADD FILTER BASED ON VIRUS TYPE

            dcc.DatePickerRange(

                id='date-range',

                min_date_allowed=min(df.publish_time),

                max_date_allowed=max(df.publish_time),

                initial_visible_month=datetime(2020, 1, 1),

                start_date=datetime(2020, 1, 1),

                end_date = datetime(2020, 1, 31)

        ),

            dcc.Graph(

            id='discover-plot',

            figure=None,

        ),

            html.P(id='selected-element')

        ]

        )

    ]),

])



@app.callback(

    Output('selected-element', 'children'),

    [Input('discover-plot', 'clickData')]

    )

def show_point_data(data_dict):

    print(data_dict)

    title = data_dict['points'][0]['customdata'][0]

    abstract = df.loc[title]['abstract']

    summary = df.loc[title]['summary']

    return [f'SUMMARY {summary}',html.Br(),html.Br(), f'ABSTRACT:{abstract}']



@app.callback(

    Output('discover-plot', 'figure'),

    [Input('dropdown-virus', 'value'),

    Input('dropdown-stage', 'value'),

    Input('dropdown-drug', 'value'),

    Input('date-range', 'start_date'),

    Input('date-range', 'end_date'),

    Input('x-axis', 'value'),

    Input('hue-axis', 'value')

    ]

    )

def discover_plot(virus, stage, drug, start, end, x_ax, hue_ax):

    start = datetime.strptime(start.split('T')[0], '%Y-%m-%d')

    end = datetime.strptime(end.split('T')[0], '%Y-%m-%d')

    data = df[(df['publish_time'] >= start) & (df['publish_time']<= end)].copy(deep=True)

    data = data[(data.virus.isin(virus)) & (data.stage.isin(stage)) & (data.drug.isin(drug))]

    df['count'] = 1

    data= data.reset_index()

    fig = px.bar(data, x=x_ax, y='count', color=hue_ax, hover_data=['title', 'publish_time_month', 'virus', 'drug', 'stage'])

    return fig



app
df.publish_time_month
data = df

df['count'] = 1

data= data.reset_index()
#If the app above does not work, we can always render the graphs without interaction

fig = px.bar(data, x='stage', y='count', color='virus', hover_data=['title', 'publish_time_month', 'virus', 'drug', 'stage'])

fig.show()
fig = px.bar(data, x='stage', y='count', color='drugs', hover_data=['title', 'publish_time_month', 'virus', 'drug', 'stage'])

fig.show()
fig = px.bar(data, x='stage', y='count', color='drugs', hover_data=['title', 'publish_time_month', 'virus', 'drug', 'stage'])

fig.show()
fig = px.scatter(pd.pivot_table(df.reset_index(), values=['title'], index=['virus', 'stage', 'publish_time_month'], aggfunc=np.count_nonzero).reset_index().rename(columns={'title':'count'}),\

                      x="publish_time_month", y="stage", color="virus", size='count',

                 hover_name="virus", title='Occurance of research tag per month per phase sized by #Occurances')

fig.show()