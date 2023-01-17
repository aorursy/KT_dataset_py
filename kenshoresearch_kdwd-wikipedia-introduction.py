from collections import Counter

import csv

import json

import os

import string



import numpy as np

import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

sns.set_context('talk')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NUM_KLAT_LINES = 5_343_564  # link_annotated_text.jsonl

NUM_PAGE_LINES = 5_362_174  # page.csv

MAX_PAGES = max(NUM_KLAT_LINES, NUM_PAGE_LINES) # change this to a smaller integer for faster runs

kdwd_path = os.path.join("/kaggle/input", "kensho-derived-wikimedia-data")
page_df = pd.read_csv(

    os.path.join(kdwd_path, "page.csv"),

    keep_default_na=False, # dont read the page title "NaN" as a null value

    nrows=MAX_PAGES, # if we want to experient with subsets

) 

page_df
def wikipedia_url_from_title(title):

    return 'https://en.wikipedia.org/wiki/{}'.format(title.replace(' ', '_'))



def wikipedia_url_from_page_id(page_id):

    return 'https://en.wikipedia.org/?curid={}'.format(page_id)



def wikidata_url_from_item_id(item_id):

    return 'https://www.wikidata.org/entity/Q{}'.format(item_id)
iloc = 0

title, page_id, item_id = page_df.iloc[iloc][['title', 'page_id', 'item_id']]

print('title={}'.format(title))

print('page_id={}'.format(page_id))

print('item_id={}'.format(item_id))
print(wikipedia_url_from_title(title))

print(wikipedia_url_from_page_id(page_id))

print(wikidata_url_from_item_id(item_id))
page_df.sort_values("views", ascending=False).head(25)
page_df["log_views"] = np.log10(page_df["views"] + 1)
fig, axes = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)

ax = axes[0]

counts, bins, patches = ax.hist(page_df['log_views'], bins=40, density=True)

ii = np.argmax(counts)

xx = (bins[ii] + bins[ii+1]) / 2

ax.axvline(xx, color='red', ls='--', alpha=0.7)

ax.axhline(0.5, color='red', ls='--', alpha=0.7)

ax.set_xlim(-0.3, 5)

ax.set_xlabel('log10 views')

ax.set_ylabel('fraction')

ax.set_title('probability distribution')



ax = axes[1]

counts, bins, patches = ax.hist(page_df['log_views'], bins=40, density=True, cumulative=True)

ax.axvline(xx, color='red', ls='--', alpha=0.7)

ax.axhline(0.5, color='red', ls='--', alpha=0.7)

ax.set_xlabel('log10 views')

ax.set_title('cumulative distribution')



fig.suptitle('Distribution of page views for {} pages'.format(page_df.shape[0]));
class KdwdLinkAnnotatedText:

    

    def __init__(self, file_path, max_pages=MAX_PAGES):

        self.file_path = file_path

        self.num_lines = NUM_KLAT_LINES

        self.max_pages = max_pages

        self.pages_to_parse = min(self.num_lines, self.max_pages)

        

    def __iter__(self):

        with open(self.file_path) as fp:

            for ii_line, line in enumerate(fp):

                if ii_line == self.pages_to_parse:

                    break

                yield json.loads(line)
file_path = os.path.join(kdwd_path, "link_annotated_text.jsonl")

klat = KdwdLinkAnnotatedText(file_path)
first_page = next(iter(klat))
print('page_id: ', first_page['page_id'])

section = first_page['sections'][0]

print('section name: ', section['name'])

print('section text: ', section['text'])

print('section link_offsets: ', section['link_offsets'])

print('section link_lengths: ', section['link_lengths'])

print('section target_page_ids: ', section['target_page_ids'])
print('anchor text -> target page id')

print('-----------------------------')

for offset, length, target_page_id in zip(

    section['link_offsets'], 

    section['link_lengths'], 

    section['target_page_ids']

):

    anchor_text = section['text'][offset: offset + length]

    print('{} -> {}'.format(anchor_text, target_page_id))
in_links = Counter()

out_links = Counter()

for page in tqdm(klat, total=klat.pages_to_parse, desc='calculating in/out links'):

    for section in page['sections']:

        in_links.update(section['target_page_ids'])

        out_links[page['page_id']] += len(section['target_page_ids'])
in_links_df = pd.DataFrame(

    in_links.most_common(),

    columns=['page_id', 'in_links'],

)
page_df = pd.merge(

    page_df, 

    in_links_df, 

    how='left').fillna(0.0)
out_links_df = pd.DataFrame(

    out_links.most_common(),

    columns=['page_id', 'out_links'],

)
page_df = pd.merge(

    page_df, 

    out_links_df, 

    how='left').fillna(0.0)
page_df['log_in_links'] = np.log10(page_df['in_links'] + 1)

page_df['log_out_links'] = np.log10(page_df['out_links'] + 1)
page_df
print(page_df['in_links'].max())

print(page_df['out_links'].max())

print(page_df['views'].max())
LIN_BINS = np.array([

    -0.5, 0.5, 10.5, 100.5, 1_000.5, 10_000.5,

    100_000.5, 1_000_000.5, 1_000_000_000.5])



BIN_NAMES = [

    '0', '1-10', '11-100', '101-1k', '1k-10k',

    '10k-100k', '100k-1M', '>1M']



lin_bins = {

    'in_links': LIN_BINS[:-1],

    'out_links': LIN_BINS[:-1],

    'views': LIN_BINS[1:],

}



bin_names = {

    'in_links': BIN_NAMES[:-1],

    'out_links': BIN_NAMES[:-1],

    'views': BIN_NAMES[1:],  

}



log_bins = {k: np.log10(v + 1) for k,v in lin_bins.items()}
for key, bins in log_bins.items():

    page_df['{}_digi'.format(key)] = np.digitize(page_df['log_{}'.format(key)], bins=bins)

    page_df['{}_bin_name'.format(key)] = page_df['{}_digi'.format(key)].apply(lambda x: bin_names[key][x-1])
page_df
def facetgrid_view_links(x_key, y_key, page_df, hist_or_box):

    if hist_or_box not in ('hist', 'box'):

        raise ValueError()

    

    x_col = '{}_bin_name'.format(x_key)

    y_col = 'log_{}'.format(y_key)    



    grpagg_df = page_df.groupby([x_col])[y_col].agg(['mean', 'median', 'size'])

    grpagg_df = grpagg_df.loc[bin_names[x_key]]

    means = grpagg_df['mean'].values

    medians = grpagg_df['median'].values

    sizes = grpagg_df['size'].values



    g = sns.FacetGrid(

        page_df, 

        col=x_col, 

        height=5, 

        aspect=0.4,

        col_order=bin_names[x_key]

    )



    bins = np.linspace(0, 7, 31)

    if hist_or_box == 'hist':

        g.map(

            sns.distplot, y_col, vertical=True, bins=bins, 

            kde=False, hist_kws={'log': False, 'density': True, 'alpha': 0.9})

    elif hist_or_box == 'box':

        g.map(sns.boxplot, y_col, orient='v')

        

    g.set_titles("")

    g.fig.subplots_adjust(wspace=0.1)

    for iax, (ax, level) in enumerate(zip(g.axes.flat, bin_names[x_key])):

        ax.axhline(y=medians[iax], color='red', ls='-', lw=1.0)

        ax.axhline(y=means[iax], color='red', ls='--', lw=1.0)

        

        if hist_or_box == 'hist':

            ax.text(1e-1, 6.2, 'n={}'.format(sizes[iax]), fontsize=14, weight='bold', color='red')

            ax.set_xlabel(None)

        elif hist_or_box == 'box':

            ax.set_xlabel(bin_names[x_key][iax])



        if iax==0:

            ax.set_ylabel('log {}'.format(y_key.replace('_', '-')))

    

    g.set(ylim=(-0.2, 7))

    g.set(xticklabels=[])

    

    if hist_or_box == 'hist':

        g.set(xlim=(0, 1.5))

        plt.suptitle('PDFs and boxplots: log {} vs {} bins'.format(y_key.replace('_', '-'), x_key.replace('_', '-')))

    elif hist_or_box == 'box':   

        g.fig.text(0.5, 0.0, s=x_key.replace('_', '-'))

        

    g.fig.subplots_adjust(bottom=0.2)
x_key = 'views'

y_key = 'in_links'

facetgrid_view_links(x_key, y_key, page_df, 'hist')

facetgrid_view_links(x_key, y_key, page_df, 'box')



x_key = 'views'

y_key = 'out_links'

facetgrid_view_links(x_key, y_key, page_df, 'hist')

facetgrid_view_links(x_key, y_key, page_df, 'box')



x_key = 'in_links'

y_key = 'views'

facetgrid_view_links(x_key, y_key, page_df, 'hist')

facetgrid_view_links(x_key, y_key, page_df, 'box')



x_key = 'out_links'

y_key = 'views'

facetgrid_view_links(x_key, y_key, page_df, 'hist')

facetgrid_view_links(x_key, y_key, page_df, 'box')
page_df[page_df['out_links']==0].sort_values('views', ascending=False).head(25)
page_df[page_df['in_links']==0].sort_values('views', ascending=False).head(25)
page_df[page_df['views']<=10].sort_values('in_links', ascending=False).head(25)
page_df[page_df['views']<=10].sort_values('out_links', ascending=False).head(25)
table = str.maketrans('', '', string.punctuation)

def tokenize(text):

    tokens = [tok.lower().strip() for tok in text.split()]

    tokens = [tok.translate(table) for tok in tokens]

    return tokens
unigrams = Counter()

words_per_section = []

for page in tqdm(

    klat, total=min(klat.num_lines, klat.max_pages), 

    desc='iterating over page text'

):

    for section in page['sections']:

        tokens = tokenize(section['text'])

        unigrams.update(tokens)

        words_per_section.append(len(tokens))

        # stop after intro section

        break

print('num tokens= {}'.format(sum(unigrams.values())))

print('unique tokens= {}'.format(len(unigrams)))
def filter_unigrams(unigrams, min_count):

    """remove tokens that dont occur at least `min_count` times"""

    tokens_to_filter = [tok for tok, count in unigrams.items() if count < min_count]

    for tok in tokens_to_filter:

        del unigrams[tok]

    return unigrams
min_count = 5

unigrams = filter_unigrams(unigrams, min_count)

print('num tokens= {}'.format(sum(unigrams.values())))

print('unique tokens= {}'.format(len(unigrams)))
unigrams_df = pd.DataFrame(unigrams.most_common(), columns=['token', 'count'])
unigrams_df
num_rows = unigrams_df.shape[0]

ii_rows_logs = np.linspace(1, np.log10(num_rows-1), 34)

ii_rows = [0, 1, 3, 7] + [int(el) for el in 10**ii_rows_logs]

rows = unigrams_df.iloc[ii_rows, :]

indexs = np.log10(rows.index.values + 1)

counts = np.log10(rows['count'].values + 1)

tokens = rows['token']



fig, ax = plt.subplots(figsize=(14,12))

ax.scatter(indexs, counts)

for token, index, count in zip(tokens, indexs, counts):

    ax.text(index + 0.05, count + 0.05, token, fontsize=12)

ax.set_xlim(-0.2, 6.5)

ax.set_xlabel('log10 rank')

ax.set_ylabel('log10 count')

ax.set_title('Zipf style plot for unigrams');
xx = np.log10(np.array(words_per_section) + 1)



fig, axes = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)

ax = axes[0]

counts, bins, patches = ax.hist(xx, bins=40, density=True)

ii = np.argmax(counts)

xx_max = (bins[ii] + bins[ii+1]) / 2

ax.axvline(xx_max, color='red', ls='--', alpha=0.7)

ax.axhline(0.5, color='red', ls='--', alpha=0.7)

ax.set_xlabel('log10 tokens/section')

ax.set_ylabel('fraction')

ax.set_title('probability distribution')

ax.set_xlim(0.3, 3.8)



ax = axes[1]

counts, bins, patches = ax.hist(xx, bins=40, density=True, cumulative=True)

ax.axvline(xx_max, color='red', ls='--', alpha=0.7)

ax.axhline(0.5, color='red', ls='--', alpha=0.7)

ax.set_xlabel('log10 tokens/section')

ax.set_title('cumulative distribution')



fig.suptitle('Distribution of tokens/section for {} pages'.format(len(words_per_section)));