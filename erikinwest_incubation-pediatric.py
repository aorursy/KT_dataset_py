import numpy as np

import pandas as pd

import os

import re



import seaborn as sns

from datetime import datetime as dt



from support_funs_incubation import stopifnot, uwords, idx_find, find_beside, ljoin, sentence_find, record_vals



!pip install ansicolors

# Takes a tuple (list(idx), sentence) and will print in red anything in the index

def color_printer(idx_sentence):

    indices = idx_sentence[0]

    sentence = idx_sentence[1]

    mat = np.zeros([2 * len(indices) + 1, 2], dtype=int)

    for ii, idx in enumerate(indices):

        ri = 2 * ii + 1

        mat[ri - 1, 1] = idx[0]

        mat[ri, :] = idx

        mat[ri + 1, 0] = idx[1]

        if ii + 1 == len(indices):

            mat[ri + 1, 1] = len(sentence)

    output = ''

    for ii in range(mat.shape[0]):

        if ii % 2 == 0:

            output = output + sentence[mat[ii, 0]:mat[ii, 1]]

        else:

            output = output + red(sentence[mat[ii, 0]:mat[ii, 1]])

    output = output.replace('\n', '')

    print(output)



from colors import red, black, white  # ansicolors



dir_base = os.getcwd()

dir_data = os.path.join(dir_base,'..','input','incubation')

# load data

df = pd.read_csv(os.path.join(dir_data, 'df_txt.csv'))

df['date'] = pd.to_datetime(df.date)

print(df.shape)



# remove prefix from some abstracts: publically funded repositories.... etc

pref = 'COVID-19 resource centre remains active.'

for ii, aa in enumerate(df.abstract):

    if isinstance(aa, float):  # nan

        continue

    hit = re.search(pref, aa)

    if hit:

        df.abstract.iloc[ii] = aa[hit.span()[1] + 1:]
# Find ways in which covid and ncov are referred to

regex_ncov = r'(20)?19(\-)?ncov|ncov(\-)?(20)?19'

regex_covid = r'covid(\-)?(20)?19'



# row indices

idx_covid_abs = np.where(idx_find(df.abstract, regex_covid))[0]

idx_ncov_abs = np.where(idx_find(df.abstract, regex_ncov))[0]

idx_union_abs = np.union1d(idx_covid_abs, idx_ncov_abs)



di_regex = {'covid': regex_covid, 'ncov': regex_ncov}

di_idx = {'covid': idx_covid_abs, 'ncov': idx_ncov_abs}



print('%i possible "covid" articles (using abstract)\n'

      '%i possible nCoV articles (using abstract)\n'

      'Union: %i, interection: %i' %

      (len(idx_covid_abs), len(idx_ncov_abs), len(idx_union_abs),

       len(np.intersect1d(idx_covid_abs, idx_ncov_abs))))



dfmt = '%B %d, %Y'

date_ncov_min = df.date.iloc[idx_ncov_abs].min().strftime(dfmt)

date_ncov_max = df.date.iloc[idx_ncov_abs].max().strftime(dfmt)

date_covid_min = df.date.iloc[idx_covid_abs].min().strftime(dfmt)

date_covid_max = df.date.iloc[idx_covid_abs].max().strftime(dfmt)



print('First and last nCoV article: %s & %s\n'

      'First and last covid-19 article: %s & %s' %

      (date_ncov_min, date_ncov_max, date_covid_min, date_covid_max))



holder = []

for term in di_regex:

    regex = di_regex[term]

    idx = di_idx[term]

    dat_abstract = uwords(df.abstract.iloc[idx], regex).assign(doc='abstract')

    dat_txt = uwords(df.txt.iloc[idx], regex).assign(doc='txt')

    dat = pd.concat([dat_abstract, dat_txt])

    dat = dat.groupby('term').n.sum().reset_index()

    dat.insert(0, 'tt', term)

    holder.append(dat)

df_term = pd.concat(holder).reset_index(drop=True)

# Term usage

print(df_term)
pat_peds = r'infant|child|pediatric|age\<'



idx_incubation = []

idx_peds = []

for ii in idx_union_abs:

    abs, txt = df.abstract[ii], df.txt[ii]

    corpus = abs + '. ' + txt

    if re.search(r'incubation', corpus, re.IGNORECASE) is not None:

        idx_incubation.append(ii)

    if re.search(pat_peds, corpus, re.IGNORECASE) is not None:

        idx_peds.append(ii)

idx_incubation_peds = np.intersect1d(idx_incubation, idx_peds)



print('%i incubation articles, with %i pediatric articles, %i overlap' %

      (len(idx_incubation), len(idx_peds), len(idx_incubation_peds)))



# What is the most common word to appear before/after incubation?

holder_l, holder_r = [], []

for ii in idx_incubation:

    abs, txt = df.abstract[ii], df.txt[ii]

    corpus = abs + '. ' + txt

    rterm = find_beside(corpus, 'incubation', tt='right')

    lterm = find_beside(corpus, 'incubation', tt='left')

    holder_r.append(rterm)

    holder_l.append(lterm)



dat_suffix = pd.Series(ljoin(holder_r)).str.lower().value_counts().reset_index().rename(

    columns={0: 'n', 'index': 'suffix'})

dat_prefix = pd.Series(ljoin(holder_l)).str.lower().value_counts().reset_index().rename(

    columns={0: 'n', 'index': 'suffix'})

print(dat_suffix.head(50))

print(dat_prefix.head(50))



suffix = ['period', 'time', 'distribution', 'duration', 'interval', 'rate', 'mean', 'median', 'estimation']

suffix = [z + r'(s)?' for z in suffix]

pat_incubation = [r'incubation\s'+z for z in suffix]
do_run = False

if do_run:

    keepers = []

    for jj, ii in enumerate(idx_incubation):

        abs, txt = df.abstract[ii], df.txt[ii]

        corpus = abs + '. ' + txt

        idx_sentences = sentence_find(corpus, pat_incubation)

        if len(idx_sentences) > 0:

            try:

                dd = df.loc[ii,'date'].strftime('%B %d, %Y')

            except:

                dd = 'NaN'

            print('---- Title: %s, date: %s, index: %i (%i of %i) ----' %

                  (df.loc[ii, 'title'], dd , ii,jj+1,len(idx_incubation)))

            tmp = record_vals(idx_sentences)

            dat = pd.DataFrame(tmp,columns=['pos','txt']).assign(idx = ii)

            keepers.append(dat)

    dat_sentences = pd.concat(keepers)

    dat_sentences = dat_sentences[['idx','pos','txt']]

    dat_sentences['txt'] = dat_sentences.txt.str.replace('\n','')

    dat_sentences = df.iloc[idx_incubation][['source','title','doi','date']].rename_axis('idx').reset_index().merge(

                    dat_sentences,on='idx',how='right')

    dat_sentences.to_csv(os.path.join(dir_output,'sentence_flag.csv'),index=False)
df_moments = pd.read_csv(os.path.join(dir_data,'sentence_flag.csv'))

df_txt = df_moments[['title','pos','txt']].copy()

df_moments.drop(columns = ['pos','txt'],inplace=True)

df_moments['date'] = pd.to_datetime(df_moments.date)

moments = df_moments.moments.str.split('\;',expand=True).reset_index().melt('index')

moments = moments[moments.value.notnull()].reset_index(drop=True).drop(columns='variable')

tmp = moments.value.str.split('\=',expand=True)

moments = moments.drop(columns='value').assign(moment=tmp.iloc[:,0], val=tmp.iloc[:,1].astype(float))

df_moments = df_moments.drop(columns='moments').reset_index().merge(moments,on='index',how='right').drop(columns='index')

# Print off key sentences

print('A total of %i unique studies' % (df_moments.title.unique().shape[0]) )

print('\n\n')

for ii, rr in df_txt.iterrows():

    print('----- Article: %s -----' % rr['title'] )

    idx = [int(z) for z in re.findall(r'\d+', rr['pos'])]

    idx = np.array(idx).reshape([int(len(idx) / 2), 2])

    idx = [tuple(idx[i]) for i in range(idx.shape[0])]

    sentence = rr['txt']

    idx_sentence = (idx,sentence)

    color_printer(idx_sentence)

    print('\n')
di_moments = {'lb':'Lower-bound','ub':'Upper-bound','mu':'Mean','med':'Median',

              'q2':'25th percentile','q3':'75th percentile'}

# Plot the moments over time

g = sns.FacetGrid(data=df_moments.assign(moment=lambda x: x.moment.map(di_moments)),

                  col='moment',col_wrap=3,sharex=True,sharey=False,height=4,aspect=1)

g.map(sns.lineplot,'date','val',ci=None)

g.map(sns.scatterplot,'date','val')

g.set_xlabels('');g.set_ylabels('Days')

g.fig.suptitle(t='Figure: Estimate of Incubation period moments over time',size=16,weight='bold')

g.fig.subplots_adjust(top=0.85)

for ax in g.axes.flat:

    ax.set_title(ax.title._text.replace('moment = ', ''))



# dates = [dt.strftime(dt.strptime(z,'%Y-%m-%d'),'%b-%d, %y') for z in dates]

xticks = [737425., 737439., 737456., 737470., 737485., 737499.]

lbls = ['Jan-01, 20', 'Jan-15, 20', 'Feb-01, 20', 'Feb-15, 20', 'Mar-01, 20', 'Mar-15, 20']

g.set_xticklabels(rotation=45,labels=lbls)

g.set(xticks = xticks)
ave = df_moments.groupby('moment').val.mean().reset_index().rename(columns={'moment':'Moment','val':'Average'}).assign(Moment=lambda x: x.Moment.map(di_moments))

print(np.round(ave,1))
# Get the index

df_match = df_txt.merge(df,on='title',how='left').rename(columns={'txt_x':'sentence','txt_y':'txt_full'})



for jj, rr in df_match.iterrows():

    try:

        dd = rr['date'].strftime('%B %d, %Y')

    except:

        dd = 'NaN'

    corpus = rr['abstract'] + '. ' + rr['txt_full']

    peds_sentences = sentence_find(corpus, pat_peds)

    incubation_sentences = sentence_find(corpus, pat_incubation)

    if len(peds_sentences) > 0 and len(incubation_sentences) > 0:

        print('---- Title: %s, date: %s (%i of %i) ----' %

              (rr['title'], dd, jj+1, df_match.shape[0]))

        for ii_ss in peds_sentences + incubation_sentences:

            color_printer(ii_ss)

        print('\n')
from PIL import Image

from matplotlib import pyplot as plt

image = Image.open(os.path.join(dir_data,"age_incubation.png"))

fig, ax = plt.subplots(figsize=(18,9))

ax.imshow(image)

fig.suptitle("Figure 3: from (Han 2020) ", fontsize=18,weight='bold')

fig.subplots_adjust(top=1.1)