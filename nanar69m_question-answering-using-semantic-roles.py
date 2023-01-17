import pandas as pd

from spacy import displacy

import re

from termcolor import colored

from collections import Counter
new_metadata = pd.read_pickle('../input/bert-question-abstract-similarity/metadata_2020.pkl')
new_metadata.head()
def color(text, to_color):

    colored_text = text

    for tc, color in to_color.items():

        colored_text = re.sub(tc, colored(tc, color), colored_text, flags=re.IGNORECASE)

    return colored_text
COVID = set(['SARS-CoV2', 'SARS-CoV-2', 'COVID19', 'COVID-19', 'new coronavirus'])

# TARGET = set(['vaccin', 'treatment', 'drug', 'therapeutic'])

TARGET = set(['transmission', 'incubation', 'environment'])

# TARGET = set(['risk', 'smok', 'pregnan', 'socio'])

# TARGET = set(['surface', ])

INSIGHT = set(['associat', 'caus', 'found', 'report', 'show', 'conclud', 'confirm', 'identif', 'develop', 'estimat'])



def has_covid(sent):

    return any(re.search(t, sent, flags=re.IGNORECASE) for t in COVID)

def has_target(sent): 

    return any(re.search(t, sent, flags=re.IGNORECASE) for t in TARGET)

def has_insight(sent):

    return any(re.search(t, sent, flags=re.IGNORECASE) for t in INSIGHT)



to_color = {}

for t in COVID: to_color[t] = 'red'

for t in TARGET: to_color[t] = 'green'

for t in INSIGHT: to_color[t] = 'blue'
new_metadata.abstract.str.contains('new coronavirus').sum()
def print_baseline(target_fn):

    to_print = []

    for idx, row in new_metadata.sort_values(by='publish_time', ascending=False).iterrows():

        for sent in row.abstract_spacy.sents:

            tokens = [t.text for t in sent]

            if has_covid(sent.text) and target_fn(sent.text) and has_insight(sent.text):

                to_print.append(

                    f"{colored('title   :', 'yellow')} [{row.publish_time.date()}] {colored(row.title, 'cyan')}\n"

                    f"{colored('insight :', 'yellow')} {color(sent.text, to_color)}\n\n"

                )

    print(f"{len(to_print)} results:\n\n{''.join(to_print)}")
print_baseline(has_target)
verbs = [srl_pred['verb'] for srl_doc in new_metadata.srl for srl_sent in srl_doc for srl_pred in srl_sent['verbs']]

Counter(verbs).most_common(100)
def get_srl_tokens(tokens, tags):

    assert len(tokens) == len(tags)

    return [token for token, tag in zip(tokens, tags) if tag != 'O']
def print_srl(target_fn):

    to_print = []

    for idx, row in new_metadata.sort_values(by='publish_time', ascending=False).iterrows():

        for srl_sent in row.srl:

            text = ' '.join(srl_sent['words'])

            for srl_pred in srl_sent['verbs']:

                srl_tokens = ' '.join(get_srl_tokens(srl_sent['words'], srl_pred['tags']))

                if has_insight(srl_pred['verb']) and target_fn(srl_tokens) and has_covid(srl_tokens):

                    to_print.append(

                        f"{colored('title   :', 'yellow')} [{row.publish_time.date()}] {colored(row.title, 'cyan')}\n"

                        f"{colored('insight :', 'yellow')} {color(srl_pred['description'], to_color)}\n\n"

                    )

    print(f"{len(to_print)} results:\n\n{''.join(to_print)}")
print_srl(has_target)