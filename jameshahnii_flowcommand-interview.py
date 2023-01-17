# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%load_ext autoreload

%autoreload 2
import pandas as pd

import spacy

from spacy.displacy.render import EntityRenderer

from IPython.core.display import display, HTML
def custom_render(doc, df, column, options={}, page=False, minify=False, idx=0):

    renderer, converter = EntityRenderer, parse_custom_ents

    renderer = renderer(options=options)

    parsed = [converter(doc, df=df, idx=idx, column=column)]

    html = renderer.render(parsed, page=page, minify=minify).strip()  

    return display(HTML(html))



def parse_custom_ents(doc, df, idx, column):

    

    if column in df.columns:

        entities = df[column][idx]

        ents = [{'start': ent[1], 'end': ent[2], 'label': ent[3]} 

               for ent in entities]

    else:

        ents = [{'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_}

                for ent in doc.ents]

    return {'text': doc.text, 'ents': ents, 'title': None}



def render_entities(idx, df, options={}, column='named_ents'):

    text = df['text'][idx]

    custom_render(nlp(text), df=df, column=column, options=options, idx=idx)
options = {

    'colors': {'COMPOUND': '#FE6BFE', 'PROPN': '#18CFE6', 'NOUN': '#18CFE6', 'NP': '#1EECA6', 'NPCE': '#FF8800'}

}
pd.set_option('display.max_rows', 25)

pd.options.mode.chained_assignment = None
nlp = spacy.load('en_core_web_sm')
PATH = '../input/'
!ls {PATH}
file = '../input/flowcommandtext/flowcommand_text.csv'

df = pd.read_csv(f'{PATH}{file}')



mini_df = df[:25]

mini_df.index = pd.RangeIndex(len(mini_df.index))



# df = mini_df



display(df)
lower = lambda x: x.lower()
df = pd.DataFrame(df['text'].apply(lower))

df.columns = ['text']

display(df)
def extract_named_ents(text):

    return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]



def add_named_ents(df):

    df['named_ents'] = df['text'].apply(extract_named_ents)

    

add_named_ents(df)

display(df)
column = 'named_ents'

render_entities(0, df, options=options, column=column)
def extract_nouns(text):

    keep_pos = ['PROPN', 'NOUN']

    return [(tok.text, tok.idx, tok.idx+len(tok.text), tok.pos_) for tok in nlp(text) if tok.pos_ in keep_pos]



def add_nouns(df):

    df['nouns'] = df['text'].apply(extract_nouns)



add_nouns(df)

display(df)
column = 'nouns'

render_entities(0, df, options=options, column=column)
def extract_named_nouns(row_series):

    ents = set()

    idxs = set()

    # remove duplicates and merge two lists

    for noun_tuple in row_series['nouns']:

        for named_ents_tuple in row_series['named_ents']:

            if noun_tuple[1] == named_ents_tuple[1]: 

                idxs.add(noun_tuple[1])

                ents.add(named_ents_tuple)

        if noun_tuple[1] not in idxs:

            ents.add(noun_tuple)

    

    return sorted(list(ents), key=lambda x: x[1])



def add_named_nouns(df):

    df['named_nouns'] = df.apply(extract_named_nouns, axis=1)
add_named_nouns(df)

display(df)
column = 'named_nouns'

render_entities(0, df, options=options, column=column)
text = "And I say to thee: That thou art Peter; and upon this rock I will build my church, and the gates of hell shall not prevail against it."



spacy.displacy.render(nlp(text), jupyter=True)
def extract_noun_phrases(text):

    return [(chunk.text, chunk.start_char, chunk.end_char, chunk.label_) for chunk in nlp(text).noun_chunks]



def add_noun_phrases(df):

    df['noun_phrases'] = df['text'].apply(extract_noun_phrases)
def visualize_noun_phrases(text):

    df = pd.DataFrame([text]) 

    df.columns = ['text']

    add_noun_phrases(df)

    column = 'noun_phrases'

    render_entities(0, df, options=options, column=column)



visualize_noun_phrases(text)
add_noun_phrases(df)

display(df)
column = 'noun_phrases'

render_entities(0, df, options=options, column=column)
def extract_compounds(text):

    """Extract compound noun phrases with beginning and end idxs. 

    

    Keyword arguments:

    text -- the actual text source from which to extract entities

    

    """

    comp_idx = 0

    compound = []

    compound_nps = []

    tok_idx = 0

    for idx, tok in enumerate(nlp(text)):

        if tok.dep_ == 'compound':



            # capture hyphenated compounds

            children = ''.join([c.text for c in tok.children])

            if '-' in children:

                compound.append(''.join([children, tok.text]))

            else:

                compound.append(tok.text)



            # remember starting index of first child in compound or word

            try:

                tok_idx = [c for c in tok.children][0].idx

            except IndexError:

                if len(compound) == 1:

                    tok_idx = tok.idx

            comp_idx = tok.i



        # append the last word in a compound phrase

        if tok.i - comp_idx == 1:

            compound.append(tok.text)

            if len(compound) > 1: 

                compound = ' '.join(compound)

                compound_nps.append((compound, tok_idx, tok_idx+len(compound), 'COMPOUND'))



            # reset parameters

            tok_idx = 0 

            compound = []



    return compound_nps



def add_compounds(df):

    """Create new column in data frame with compound noun phrases.

    

    Keyword arguments:

    df -- a dataframe object

    

    """

    df['compounds'] = df['text'].apply(extract_compounds)
add_compounds(df)

display(df)
column = 'compounds'

render_entities(0, df, options=options, column=column)
def extract_comp_nouns(row_series, cols=[]):

    return {noun_tuple[0] for col in cols for noun_tuple in row_series[col]}



def add_comp_nouns(df, cols=[]):

    df['comp_nouns'] = df.apply(extract_comp_nouns, axis=1, cols=cols)



cols = ['nouns', 'compounds']

add_comp_nouns(df, cols=cols)

display(df)
# take a look at all the nouns again

column = 'named_nouns'

render_entities(0, df, options=options, column=column)
# take a look at all the compound noun phrases again

column = 'compounds'

render_entities(0, df, options=options, column=column)
# take a look at combined entities

df['comp_nouns'][0] 
def drop_duplicate_np_splits(ents):

    drop_ents = set()

    for ent in ents:

        if len(ent.split(' ')) > 1:

            for e in ent.split(' '):

                if e in ents:

                    drop_ents.add(e)

    return ents - drop_ents



def drop_single_char_nps(ents):

    return {' '.join([e for e in ent.split(' ') if not len(e) == 1]) for ent in ents}



def drop_double_char(ents):

    drop_ents = {ent for ent in ents if len(ent) < 3}

    return ents - drop_ents



def keep_alpha(ents):

    keep_char = set('-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

    drop_ents = {ent for ent in ents if not set(ent).issubset(keep_char)}

    return ents - drop_ents
!ls {PATH}
filename = './topfreqwords/freq_words.csv'

freq_words_df = pd.read_csv(f'{PATH}{filename}')

display(freq_words_df)
freq_words = freq_words_df['Word'].iloc[1:]

display(freq_words)
def remove_freq_words(ents):

    freq_words = pd.read_csv('../input/topfreqwords/freq_words.csv')['Word'].iloc[1:]

    for word in freq_words:

        try:

            ents.remove(word)

        except KeyError:

            continue # ignore the stop word if it's not in the list of abstract entities

    return ents



def add_clean_ents(df, funcs=[]):

    col = 'clean_ents'

    df[col] = df['comp_nouns']

    for f in funcs:

        df[col] = df[col].apply(f)
funcs = [drop_duplicate_np_splits, drop_double_char, keep_alpha, drop_single_char_nps, remove_freq_words]

add_clean_ents(df, funcs)

display(df)
def visualize_entities(df, idx=0):

    # store entity start and end index for visualization in dummy df

    ents = []

    abstract = df['text'][idx]

    for ent in df['clean_ents'][idx]:

        i = abstract.find(ent) # locate the index of the entity in the abstract

        ents.append((ent, i, i+len(ent), 'NPCE')) 

    ents.sort(key=lambda tup: tup[1])



    dummy_df = pd.DataFrame([abstract, ents]).T # transpose dataframe

    dummy_df.columns = ['text', 'clean_ents']

    column = 'clean_ents'

    render_entities(0, dummy_df, options=options, column=column)
visualize_entities(df, 0)