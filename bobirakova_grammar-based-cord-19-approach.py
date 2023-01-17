import pandas as pd

import plotly.express as px

import plotly.graph_objects as go



oxford_indicators = [

    "School or university closing",

    "Workplace closing",

    "Cancel public events",

    "Close public transport",

    "Public information campaigns",

    "Restrictions on internal movement",

    "International travel restrictions and control",

    "Fiscal measures",

    "Monetary measures",

    "Emergency investment in healthcare",

    "Investment in vaccines",

    "Testing policy",

    "Contact tracing"

]



# Read cached results

cached_plot_df = pd.read_csv("../input/cached-outputs/cached_plot_matches.csv", na_values='')



title = 'Barriers and enablers (RED), and implications (BLUE) of the uptake of public health measures <br>given the variation in policy responses, <i>hover over dots for top matches</i>'

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=list(cached_plot_df['count_matches'])[::-1],

    y=list(oxford_indicators)[::-1],

    marker=dict(color="crimson", size=12),

    mode="markers",

    hovertemplate = '<b>%{x}: %{y} %{text}',

    text = cached_plot_df['top_matches_uptake'],

    showlegend = False))



fig.add_trace(go.Scatter(

    x=list(cached_plot_df['count_matches_impl'])[::-1],

    y=list(oxford_indicators)[::-1],

    marker=dict(color="blue", size=12),

    mode="markers",

    hovertemplate = '<b>%{x}: %{y} %{text}',

    text = cached_plot_df['top_matches_impl'],

    showlegend = False))





fig.add_annotation



fig.update_layout(title=title,

              hoverlabel_align = 'left',

              hovermode='closest',

              xaxis_title='Counts of references in the corpus',

              yaxis_title='Policy Responses')



fig.show()
import covid19_tools as cv19

import pandas as pd

import re

from IPython.core.display import display, HTML

import html

import numpy as np

import json

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import glob

import os



METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'



# Load metadata

meta = cv19.load_metadata(METADATA_FILE)



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
!dpkg -i ../input/libgrapenlp/libgrapenlp_2.8.0-0ubuntu1_xenial_amd64.deb

!dpkg -i ../input/libgrapenlp/libgrapenlp-dev_2.8.0-0ubuntu1_xenial_amd64.deb

!pip install pygrapenlp



from collections import OrderedDict

from pygrapenlp import u_out_bound_trie_string_to_string

from pygrapenlp.grammar_engine import GrammarEngine
from ipywidgets import Image

f = open("../input/grammar/grammar.png", "rb")

image = f.read()

Image(value=image)
def parse_into_sentences(text):

    regex_list = [re.compile(r"([ (\[]%s([.1-9 ]+[A-Z]?[a-z]?)*[ )\]])" % name, re.IGNORECASE) for name in ["fig", "figure", "table"]]

    regex_list.extend([re.compile(r"%s" % term, re.IGNORECASE) for term in ["et al\.", "i\.e\.", "e\.g\.", " [1-9]\.", "\.\.\.", "\.(?=\))"]])

    regex_list.append(re.compile(r" [A-Z]\."))



    preprocessed_text = text

    #preprocessed_text

    for r in regex_list:

        preprocessed_text = r.sub("", preprocessed_text)



    sentences_list = preprocessed_text.split('.')

    

    return sentences_list
def native_results_to_python_dic(sentence, native_results):

    top_segments = OrderedDict()

    if not native_results.empty():

        top_native_result = native_results.get_elem_at(0)

        top_native_result_segments = top_native_result.ssa

        for i in range(0, top_native_result_segments.size()):

            native_segment = top_native_result_segments.get_elem_at(i)

            native_segment_label = native_segment.name

            segment_label = u_out_bound_trie_string_to_string(native_segment_label)

            segment = OrderedDict()

            segment['value'] = sentence[native_segment.begin:native_segment.end]

            segment['start'] = native_segment.begin

            segment['end'] = native_segment.end

            top_segments[segment_label] = segment

    return top_segments



def execute_phm_soc_grammar():

    base_dir = os.path.join('..', 'input', 'grammar')

    grammar_pathname = os.path.join(base_dir, 'soc_grammar.fst2')

    bin_delaf_pathname = os.path.join(base_dir, 'test_delaf.bin')

    grammar_engine = GrammarEngine(grammar_pathname, bin_delaf_pathname)



    df = pd.DataFrame([], columns=['index', 'list_phm_impl', 'list_phm_uptake', 'sentence'])



    for record_id in range(len(full_text_repr)):

        record_text = "".join([item['text'] for item in full_text_repr[record_id]['body_text']])

        record_sentences = parse_into_sentences(record_text)

        processed = 0



        for sentence in record_sentences:

            context = {}

            native_results = grammar_engine.tag(sentence, context)

            matches = native_results_to_python_dic(sentence, native_results)



            if 'list_phm_impl' in matches or 'list_phm_uptake' in matches:

                df = df.append(

                {

                    'index': record_id, 

                    'list_phm_impl': matches['list_phm_impl']['value'] if 'list_phm_impl' in matches else '', 

                    'list_phm_uptake': matches['list_phm_uptake']['value'] if 'list_phm_uptake' in matches else '',

                    'sentence': sentence



                }, ignore_index=True)

                processed += 1



        # print("record %d, processed %d sentences, found %d matches" % (record_id, len(record_sentences), processed))



    print("Processed %d documents, found %d matches" % (len(full_text_repr), df.shape[0]))

    df.to_csv('output.csv')

    return df
grammar_terms = ['public health', 'intervention', 'policy', 'policies', 'quarantine', 'lockdown', 'contact tracing', 'distancing', 'emergency']



meta, soc_ethic_counts = cv19.count_and_tag(meta,

                                               grammar_terms,

                                               'soc_ethic')

number_of_SOC_articles = len(meta[meta.tag_soc_ethic == True])

print('Number of articles is ', number_of_SOC_articles) 



print('Loading raw data ...')

metadata_filter = meta[meta.tag_soc_ethic == True] 

full_text_repr = cv19.load_full_text(metadata_filter,

                                     '../input/CORD-19-research-challenge')
def execute_phm_extract_grammar():

    base_dir = os.path.join('..', 'input', 'grammar')

    bin_delaf_pathname = os.path.join(base_dir, 'test_delaf.bin')

    grammar_pathname = os.path.join(base_dir, 'phm_list.fst2')

    grammar_engine = GrammarEngine(grammar_pathname, bin_delaf_pathname)



    df = pd.DataFrame([], columns=['index', 'list_phm', 'sentence'])



    for record_id in range(len(full_text_repr)):

        record_text = "".join([item['text'] for item in full_text_repr[record_id]['body_text']])

        record_sentences = parse_into_sentences(record_text)

        processed = 0



        for sentence in record_sentences:

            context = {}

            native_results = grammar_engine.tag(sentence, context)

            matches = native_results_to_python_dic(sentence, native_results)



            if 'list_phm' in matches:

                df = df.append(

                {

                    'index': record_id, 

                    'list_phm': matches['list_phm']['value'] if 'list_phm' in matches else '', 

                    'sentence': sentence



                }, ignore_index=True)

                processed += 1



        # print("record %d, processed %d sentences, found %d matches" % (record_id, len(record_sentences), processed))



    print("Processed %d documents, found %d matches" % (len(full_text_repr), df.shape[0]))

    

    return df
df_grammar_output = execute_phm_extract_grammar()

df_grammar_output.to_csv('grammar_output_uptake_impl.csv')
import tensorflow as tf



# download the Universal Sentence Encoder module and uncompress it to the destination folder. 

!mkdir /kaggle/working/universal_sentence_encoder/

!curl -L 'https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed' | tar -zxvC /kaggle/working/universal_sentence_encoder/
import tensorflow_hub as hub



# load the Universal Sentence Encoder module

use_embed = hub.load('/kaggle/working/universal_sentence_encoder/')
# Read cached grammar output and transform to add the paper id / SHA value

df = pd.read_csv("../input/cached-outputs/grammar_output_uptake_impl.csv", na_values='')

df = df.drop(columns=['Unnamed: 0'])

df_full_text_repr = pd.DataFrame(full_text_repr)

paper_ids = df_full_text_repr['paper_id']

df['paper_id'] = df.apply(lambda row: paper_ids[row['index']], axis=1)



df.head()
processed = pd.melt(df, id_vars=["paper_id", "sentence"], value_vars=["list_phm_impl", "list_phm_uptake"], value_name="match").dropna()

processed.set_index('paper_id')



processed['cord_uid'] = np.nan

processed['title'] = np.nan

processed['doi'] = np.nan

processed['url'] = np.nan

processed['authors'] = np.nan

processed['authors_short'] = np.nan

processed['journal'] = np.nan



for i, row in processed.iterrows():

    meta_raw = metadata_filter.loc[metadata_filter['sha'] == row['paper_id']]

    #if len(meta_raw) == 1:

    if 'title' in meta_raw:

         processed.loc[i, 'cord_uid'] = meta_raw['cord_uid'].values[0]

         processed.loc[i, 'title'] = meta_raw['title'].values[0]

         processed.loc[i, 'doi'] = meta_raw['doi'].values[0]

         processed.loc[i, 'url'] = meta_raw['url'].values[0]

         processed.loc[i, 'authors'] = meta_raw['authors'].values[0]

         processed.loc[i, 'authors_short'] = meta_raw['authors_short'].values[0]

         processed.loc[i, 'journal'] = meta_raw['journal'].values[0]



processed = processed.rename(columns={"sentence": "excerpt", "variable": "type"})

processed.head()
oxford_indicators = [

    "School or university closing",

    "Workplace closing",

    "Cancel public events",

    "Close public transport",

    "Public information campaigns",

    "Restrictions on internal movement",

    "International travel restrictions and control",

    "Fiscal measures",

    "Monetary measures",

    "Emergency investment in healthcare",

    "Investment in vaccines",

    "Testing policy",

    "Contact tracing"

]

oxford_indicators_descriptive = [

    "school university and educational facilities",

    "workplace work job",

    "events gatherings of people",

    "public transport bus train",

    "public information campaigns to raise awareness",

    "strict lockdown quarantine",

    "international travel restrictions airport",

    "economy job loss low-income policies adopted fiscal tax spending",

    "value of interest rate",

    "investment hospitals masks healthcare public health patient",

    "vaccine development",

    "testing policy",

    "contact tracing surveillance privacy"

]



oxford_indicators_embeddings = use_embed(oxford_indicators_descriptive)
CORR_THRESHOLD = 0.2

BATCH_SIZE = 100

NUM_TOP_MATCHES_TO_RETURN = 3

MAX_CHARS_TOOLTIP = 50



def extract_grammar_matches_uptake_for_public_health_measures(df_input, grammar_label, column_to_match):

    phm_uptake = df_input[df_input['type'] == grammar_label]



    count_matches = np.zeros(len(oxford_indicators))

    df_matches = []



    for idx in range(int(len(phm_uptake)/BATCH_SIZE)):

        if idx*BATCH_SIZE > len(phm_uptake):

            break

        idx_start = idx*BATCH_SIZE

        idx_end = min((idx+1)*BATCH_SIZE, len(phm_uptake))



        phm_uptake_val = phm_uptake[idx_start:idx_end]

        phm_uptake_val = phm_uptake_val.reindex()

        phm_uptake_embeddings = use_embed(phm_uptake_val[column_to_match])

        corr_uptake = np.inner(oxford_indicators_embeddings, phm_uptake_embeddings)



        for i in range(len(phm_uptake_embeddings)):

            for j in range(len(oxford_indicators_embeddings)):

                if corr_uptake[j][i] > CORR_THRESHOLD:

                    count_matches[j] += 1

                    df_matches.append([j, 

                                       corr_uptake[j][i], 

                                       phm_uptake_val.iloc[i]['paper_id'], 

                                       phm_uptake_val.iloc[i]['excerpt'],

                                       phm_uptake_val.iloc[i]['cord_uid'],

                                       phm_uptake_val.iloc[i]['title'],

                                       phm_uptake_val.iloc[i]['doi'],

                                       phm_uptake_val.iloc[i]['url'],

                                       phm_uptake_val.iloc[i]['authors'],

                                       phm_uptake_val.iloc[i]['authors_short'],

                                       phm_uptake_val.iloc[i]['journal']]

                                     )



                    # print(" (%d, %d) Found macth %f\n ---\t%s\n ---\t%s\n ---\t%s" % (i, j, corr_uptake[j][i], oxford_indicators[j], phm_uptake_val.iloc[i], phm_uptake.iloc[i]['sentence']))

        # print("processed batch %d - %d" % (idx_start, idx_end))



        del corr_uptake

        del phm_uptake_val

        del phm_uptake_embeddings

        

    return count_matches, df_matches 
count_matches_uptake, df_matches = extract_grammar_matches_uptake_for_public_health_measures(processed, 'list_phm_uptake', 'match')

df_matches_uptake = pd.DataFrame(df_matches)

df_matches_uptake.columns = ["policy_response", "match_score", "paper_id", "excerpt", "cord_uid", "title", "doi", "url", "authors", "authors_short", "journal"]
indicator_matches = df_matches_uptake

indicator_matches = df_matches_uptake[df_matches_uptake['policy_response'] == 0]

indicator_matches.sort_values(by='match_score', ascending=False, inplace=True)

print("Found %d entries corresponding to the indicator %s" % (len(indicator_matches), oxford_indicators[0]))

indicator_matches.head()
for indicator_id in range(len(oxford_indicators)): 

    indicator_matches = df_matches_uptake

    indicator_matches = indicator_matches[indicator_matches['policy_response'] == indicator_id]

    indicator_matches.sort_values(by='match_score', ascending=False, inplace=True)



    to_save = indicator_matches.drop(['paper_id'], axis=1)

    to_save.to_csv('policy_measures_indicator_%d_findings_plus_excerpts.csv' % indicator_id)

    print("Saved %d entries corresponding to the indicator %s" % (len(to_save), oxford_indicators[indicator_id]))


def generate_plot_labels(df_matches, df_input):

    res = []

    df_input = df_input.set_index('paper_id')



    for ox_ind in range(len(oxford_indicators)):

        df_ = df_matches[df_matches['policy_response'] == ox_ind]

        df_.sort_values(by='match_score', ascending=False, inplace=True)

        combined = ''

        num_included = 0

        included = []



        for p in range(len(oxford_indicators)):

            if num_included == NUM_TOP_MATCHES_TO_RETURN: 

                break

            record_id = df_['paper_id'].iloc[p]

            if record_id not in included:

                included.append(record_id)

                record_details = ""

                

                if 'title' in df_input.loc[record_id]:

                    if type(df_input.loc[record_id]['title']) == str :

                        record_details = "- \"%s\" by %s" % (df_input.loc[record_id]['title'], df_input.loc[record_id]['authors_short'])

                    else:

                        record_details = "- \"%s\" by %s" % (df_input.loc[record_id]['title'].values[0], df_input.loc[record_id]['authors_short'].values[0])

                    num_included += 1

                    

                    if len(record_details) > MAX_CHARS_TOOLTIP:

                        r = ""

                        out = [(record_details[i:i+MAX_CHARS_TOOLTIP]) for i in range(0, len(record_details), MAX_CHARS_TOOLTIP)] 

                        for o in out:

                            r = "%s<br>%s" % (r, o)

                        record_details = r[4:]

                        

                    combined = "%s<br>%s" % (combined, record_details)

                        



        res.append(combined)

        

    return res



top_matches_uptake = generate_plot_labels(df_matches_uptake, processed)
count_matches_impl, df_matches = extract_grammar_matches_uptake_for_public_health_measures(processed, 'list_phm_impl', 'match')

df_matches_impl = pd.DataFrame(df_matches)

df_matches_impl.columns = ["policy_response", "match_score", "paper_id", "excerpt", "cord_uid", "title", "doi", "url", "authors", "authors_short", "journal"]

top_matches_impl = generate_plot_labels(df_matches_impl, processed)
# Save outputs for quick access



cached_plot = pd.DataFrame()

cached_plot['count_matches'] = count_matches_uptake

cached_plot['count_matches_impl'] = count_matches_impl

cached_plot['top_matches_uptake'] = top_matches_uptake

cached_plot['top_matches_impl'] = top_matches_impl



cached_plot.to_csv('/kaggle/working/cached_plot_matches.csv', index = False)
import plotly.express as px

import plotly.graph_objects as go



title = 'Barriers, enablers (RED), and implications (BLUE) of the uptake of public health measures <br>given the variation in policy responses, <i>hover over dots for top matches</i>'

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=list(count_matches_uptake)[::-1],

    y=list(oxford_indicators)[::-1],

    marker=dict(color="crimson", size=12),

    mode="markers",

    hovertemplate = '<b>%{x}: %{y} %{text}',

    text = top_matches_uptake,

    showlegend = False))



fig.add_trace(go.Scatter(

    x=list(count_matches_impl)[::-1],

    y=list(oxford_indicators)[::-1],

    marker=dict(color="blue", size=12),

    mode="markers",

    hovertemplate = '<b>%{x}: %{y} %{text}',

    text = top_matches_impl,

    showlegend = False))





fig.add_annotation



fig.update_layout(title=title,

              hoverlabel_align = 'left',

              hovermode='closest',

              xaxis_title='Counts of references in the corpus',

              yaxis_title='Policy Responses')



fig.show()
# Clean up before transitioning into the next section of the notebook

del full_text_repr