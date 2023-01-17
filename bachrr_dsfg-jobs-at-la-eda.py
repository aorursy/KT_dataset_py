import numpy as np

import pandas as pd

from pathlib import Path

from functools import reduce

import dateutil.parser as dateparser

from datetime import datetime

import re

import pdb

from tqdm import tqdm

from collections import Counter
import spacy

from spacy.lang.en import English
import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
nlp_en = English()
Path.ls = lambda self: [fn for fn in self.glob('*')]
PATH = Path('../input/cityofla/CityofLA/')

bulletins = PATH/'Job Bulletins'
fn = bulletins.ls()[39]
with open(fn, encoding='latin-1') as f:

    print(fn.name)

    print(''.join(f.readlines()[:10]))
def new_data():

    return {

        'File Name': [],

        'Class Code': [],

        'Job Title': [],

        'Open Date': [],

        'ANNUAL SALARY': [],

        'DUTIES': [],

        'REQUIREMENTS': [],

        'WHERE TO APPLY': [],

        'APPLICATION DEADLINE': [],

        'SELECTION PROCESS': []

    }



sections = ['ANNUAL SALARY', 'DUTIES', 'REQUIREMENTS', 'WHERE TO APPLY', 'APPLICATION DEADLINE', 'SELECTION PROCESS']



def is_open_date(line):

    return line.startswith('Open Date:')



def on_open_date(data, key, value):

    data[key][-1] = dateparser.parse(value.split()[2])

    key, value = None, None

    return key, value



def is_section_begin(sections, line):

    def _startswith(n=None):

        return len(list(filter(lambda section: line.startswith(section[:n]), sections)))>0

    return (line in sections) or _startswith() or _startswith(n=-1)



def get_section_key(sections, line):

    if line in sections:

        return line

    # find appropriate key, e.g. REQUIREMENTS/MINIMUM QUALIFICATIONS -> REQUIREMENTS

    for section in sections:

        if line.startswith(section) or line.startswith(section[:-1]):

            return section



def on_section_begin(data, key, value):

    return key, value



def on_section_end(data, key, value):

    if key is not None:

        key = get_section_key(sections, key)

        data[key][-1] = value

        key, value = None, None

    return key, value



def add_row(data, sections):

    for section in sections:

        data[section].append('')



def parse_sections(f, data, sections):

    add_row(data, sections + ['File Name', 'Job Title', 'Class Code', 'Open Date'])

    data['File Name'][-1] = fn.name

    key, value = None, None

    for index, line in enumerate(f):

        line = line.strip()

        if index == 0:

            data['Job Title'][-1] = line.strip()

        elif line.startswith('Class Code:'):

            data['Class Code'][-1] = line.split()[2]

        elif is_open_date(line):

            key, value = on_open_date(data, 'Open Date', line)    # handle Open Date section

        elif is_section_begin(sections, line):

            key, value = on_section_end(data, key, value)         # end previous section

            key, value = on_section_begin(data, line, '') # begin current section

        elif key is not None:

            value += line  

    on_section_end(data, key, value)                              # end last section
data = new_data()

with open(fn, encoding='latin-1') as f:

    parse_sections(f, data, sections)
assert data['File Name'][0] == 'POLICE COMMANDER 2251 092917.txt'

assert data['Class Code'][0] == '2251'

assert data['Job Title'][0] == 'POLICE COMMANDER'

assert data['Open Date'][0].strftime('%m-%d-%Y') == '09-29-2017'
pd.DataFrame(data)
data = new_data()

for fn in tqdm(bulletins.ls()):

    with open(fn, encoding='latin-1') as f:

        parse_sections(f, data, sections)
df = pd.DataFrame(data)

df.head()
df['DUTIES'].str.len().hist()
df['REQUIREMENTS'].str.len().hist()
df['WHERE TO APPLY'].str.len().hist()
df['ANNUAL SALARY'].values[10]
def get_minmax_salary(s):

    if not s or len(s) == 0:

        return 0, 0

    salary_regex = r'\$[0-9]+,?[0-9]+\.?[0-9]*'

    salaries = re.findall( salary_regex, s)

    if len(salaries)==0:

        return 0, 0

    salaries = list(map(lambda s: float(s.replace('$', '').replace(',', '')), salaries))

    return min(salaries), max(salaries)
salary_minmax = list(map(get_minmax_salary, df['ANNUAL SALARY'].values))

salary_ranges = [max_salary-min_salary for (min_salary, max_salary) in salary_minmax]
# plot a histogram for the ranges

plt.hist(salary_ranges)
top10 = sorted(salary_ranges, reverse=True)[:10]; top10

#salary_ranges.sort(reverse=True); salary_ranges[:10]
mask = [x in top10 for x in salary_ranges]

df[mask]
df.groupby(df['Open Date'].dt.year).agg({'File Name': 'count'}).plot(figsize=(8, 8), marker='*')
df.groupby(df['Open Date'].dt.month).agg({'File Name': 'count'}).plot(figsize=(8, 8), kind='bar')
doc = nlp_en(' '.join(df['Job Title'].values).lower())
# all tokens that arent stop words or punctuations

words = [token.text for token in doc if not (token.is_stop or token.is_punct or token.is_space)]

# noun tokens that arent stop words or punctuations

nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]
ax = sns.barplot(y='word', x='frequency', data=pd.DataFrame(Counter(words).most_common(20), columns=['word', 'frequency']))
nlp_en.add_pipe(nlp_en.create_pipe('sentencizer')) # enable spacy to split sentences
#doc2 = list(map(lambda text: [sent for sent in nlp_en(text.lower()).sents], df['Job Title'].values))

doc2 = [sent for sent in nlp_en(df['REQUIREMENTS'].values[0].lower()).sents]
sents_len = list(map(lambda text: [len(sent) for sent in nlp_en(text.lower()).sents], df['REQUIREMENTS'].values))
fig = plt.figure(figsize=(10, 10))

start, end = 0, 10

for i in range(start, end):

    plt.plot(sents_len[i], label=('%2d' % i))

plt.legend()
plt.hist([len(sl) for sl in sents_len])
# pad the array of sentences length

def right_pad(a, m):

    while len(a)<m: a.append(0)

    return a

max_len = min(30, max([len(sl) for sl in sents_len]))



trimed_sents_len = [sl if len(sl)<=max_len else sl[:max_len] for sl in sents_len]



padded_sents_len = [sl if len(sl)==max_len else right_pad(sl, max_len) for sl in trimed_sents_len]
arr = np.array(padded_sents_len)



plt.figure(figsize=(30, 30))

ax = plt.gca()

im = ax.imshow(arr, cmap='viridis')



divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.2)



plt.colorbar(im, cax=cax)

plt.show()
nlp_en_web = spacy.load('en_core_web_sm')
requirements_doc = nlp_en_web(df['REQUIREMENTS'].values[0].lower())

requirements_tag_df = pd.DataFrame([(token.text, token.pos_) for token in requirements_doc], columns=['word', 'tag'])

requirements_tag_df.head()
requirements_tag_df['tag'].unique()
requirements_tag_df[requirements_tag_df['tag']=='PRON']
duties_doc = nlp_en_web(df['DUTIES'].values[0].lower())

duties_tag_df = pd.DataFrame([(token.text, token.pos_) for token in duties_doc], columns=['word', 'tag'])

duties_tag_df.head()
duties_tag_df['tag'].unique() # no pronoun use