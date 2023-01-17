# Imports
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import gaussian_kde
from scipy import stats
import matplotlib.mlab as mlab
from nltk.tokenize import word_tokenize
import powerlaw
import pylab
pylab.rcParams['xtick.major.pad']='8'
pylab.rcParams['ytick.major.pad']='8'
#pylab.rcParams['font.sans-serif']='Arial'

from matplotlib import rc
rc('font', family='sans-serif')
rc('font', size=10.0)
rc('text', usetex=False)


from matplotlib.font_manager import FontProperties

panel_label_font = FontProperties().copy()
panel_label_font.set_weight("bold")
panel_label_font.set_size(12.0)
panel_label_font.set_family("sans-serif")
# Functions

def get_substring(s):
    # This function takes a sentence as input and returns a tuple with the connective, 
    # the condition and both the connective and condition as a whole.
    slice_connective = s.sentence_text[s.begin_connective:s.end_connective].strip()
    slice_condition = s.sentence_text[s.begin_condition:s.end_condition].strip()
    condition = s.sentence_text[s.begin_connective:s.end_condition].strip()
    return pd.Series(dict(connective=slice_connective, condition=slice_condition, full_condition=condition))

def plot_box_whisker(values, labels, threshold, box_width):
    # This function plots a box and whisker plot.
    fig = plt.figure(1, figsize=(6,6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(values, patch_artist=True, widths=[box_width for i in range(len(values))])
    ax.set_xticklabels(labels)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_ylim(0,threshold * 1.4)
    ax.axhline(y=threshold, color='r') 
    plt.show()
    
def ccdf_plot(lang, data):
    plt.figure()
    fit = powerlaw.Fit(data, discrete=True)
    ####
    fit.distribution_compare('power_law', 'lognormal')
    fig = fit.plot_ccdf(linewidth=3, label='Connectives')
    fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power law')
    fit.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label='Lognormal')
    fit.exponential.plot_ccdf(ax=fig, color='b', linestyle='--', label='Exponential')
    ####
    fig.set_title('CCDF plot for '+lang)
    fig.set_ylabel(u"CCDF")
    fig.set_xlabel("Frequency")
    fig.set_ylim(0.01,1)
    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles, labels, loc=3)
    
def likelihood_ratio_test(lang, results_fit, dist1, dist2):
    R, p = results_fit.distribution_compare(dist1, dist2)
    return [lang, dist1, dist2, R, p]

def scatter_with_gaussian(dim_red, lang, points):
    x, y = points[:,0], points[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(dim_red + ' for '+lang)
    cax = ax.scatter(x, y, c=z, s=30, edgecolor='')
    fig.colorbar(cax)

    plt.show()
# load data
df_conds = pd.read_csv('../input/conditions.csv')
df_sents = pd.read_csv('../input/sentences.csv')
# creating combinated dataframes
conds_sents = pd.merge(df_conds, df_sents, on='sentence_uuid', how='left', suffixes=['','_sent'])
conditions = conds_sents.apply(get_substring, axis=1)
conds_sents['connective'] = conditions.connective
conds_sents['condition'] = conditions.condition
conds_sents['full_condition'] = conditions.full_condition
# getting languages
languages = df_sents['language'].unique().tolist()
# create the table of summary
conds_sents.groupby(['language', 'domain']).connective.describe()
conds_sents_count = conds_sents.copy()
conds_sents_count['num_tokens'] = conds_sents_count.full_condition.apply(word_tokenize).str.len()
num_tokens_conds = []
for lang in languages:
    num_tokens_conds.append(conds_sents_count[(conds_sents_count['language']==lang)].num_tokens)
plot_box_whisker(num_tokens_conds, languages, 50, 0.5)
df_sents['num_tokens'] = df_sents[df_sents['labelled']].sentence_text.apply(word_tokenize).str.len()
num_tokens_per_lang = []
for lang in languages:
    num_tokens_per_lang.append(df_sents[(df_sents['language']==lang) & (df_sents['labelled'])].num_tokens)
plot_box_whisker(num_tokens_per_lang, languages, 100, 0.5)
list_df_conn = []
for lang in languages:
    df_conn = conds_sents[conds_sents['language'] == lang].copy()
    df_conn['connective'] = df_conn.connective.str.lower()
    df_conn = df_conn.groupby('connective').agg('count').reset_index()
    df_conn = df_conn[['connective', 'condition']].sort_values(by='condition', ascending=False).reset_index(drop=True).copy()
    df_conn.rename(index=str, columns={"condition" : "freq"}, inplace=True)
    list_df_conn.append(df_conn)
    conn_quarter = round(df_conn.shape[0]/4)
    print('Connectives for lang: '+lang)
    print('Top 5')
    print(df_conn.iloc[0:6])
    print('Five around the 75-th percentile')
    print(df_conn.iloc[conn_quarter-3:conn_quarter+3])
# plot of ccdf
for (df_conn, lang) in zip(list_df_conn, languages):
    data = df_conn.freq.values
    ccdf_plot(lang, data)
df_powerlaw_rp = pd.DataFrame(columns = ['Lang', 'Dist1', 'Dist2', 'R', 'p-value'])
for (df_conn, lang) in zip(list_df_conn,languages):
    results_fit = powerlaw.Fit(df_conn.freq.values, discrete=True)
    df_powerlaw_rp.loc[len(df_powerlaw_rp)] = likelihood_ratio_test(lang, results_fit, 'power_law', 'lognormal')
    df_powerlaw_rp.loc[len(df_powerlaw_rp)] = likelihood_ratio_test(lang, results_fit, 'power_law', 'exponential')
    df_powerlaw_rp.loc[len(df_powerlaw_rp)] = likelihood_ratio_test(lang, results_fit, 'lognormal', 'exponential')    
    
print(df_powerlaw_rp)
# number of words of the vectorisation
print('Number of words of the vectorisation')
words_per_lang = []
for lang in languages:
    words = conds_sents.loc[conds_sents.language == lang].full_condition.str.lower().values.ravel()
    vectorizer = TfidfVectorizer()
    words = vectorizer.fit_transform(words)
    words_per_lang.append(words)
    print(lang,':',words.shape[1])
# tsvd
for (words, lang) in zip(words_per_lang, languages):
    svd = TruncatedSVD(n_components=2,random_state=0)
    points = svd.fit_transform(words)
    scatter_with_gaussian('tsvd', lang, points)
# isomap
for (words, lang) in zip(words_per_lang, languages):
    isomap = Isomap(n_neighbors=5, n_components=2)
    points = isomap.fit_transform(words)
    scatter_with_gaussian('isomap',lang, points)