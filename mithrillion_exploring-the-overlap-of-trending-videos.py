# %load /home/mithrillion/default_imports.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
%matplotlib inline
import re
from collections import Counter
from functools import partial
from itertools import product, combinations
from datetime import datetime
# load data
country_list = ["us", "ca", "de", "fr", "gb"]

def read_country_dat(code):
    dat = pd.read_csv("../input/{0}videos.csv".format(code.upper()), parse_dates=['publish_time'])
    dat = dat[dat['trending_date'].apply(lambda x: re.match(r'\d{2}\.\d{2}\.\d{2}', x) is not None)]
    dat['trending_date'] = dat['trending_date'].apply(lambda x: datetime.strptime(x, "%y.%d.%m"))
    return dat

dat = {code: read_country_dat(code) for code in country_list}
def read_country_cat(code):
    cat = pd.read_json("../input/{0}_category_id.json".format(code.upper()), orient='columns')
    cat_exp = pd.DataFrame(list(cat['items'].values))
    cat_exp2 = pd.DataFrame(list(cat_exp['snippet'].values))
    cat_flat = pd.concat([cat_exp, cat_exp2], axis=1)
    return cat_flat

cat = {code: read_country_cat(code) for code in country_list}
def cat_merge(x, y, **kwargs):
    return pd.merge(x, y, on='id', how='outer', **kwargs)
u = cat_merge(cat['us'][['id', 'title']], cat['ca'][['id', 'title']], suffixes=('_us', '_ca'))
u = cat_merge(u, cat['de'][['id', 'title']], suffixes=('', '_de'))
u = cat_merge(u, cat['fr'][['id', 'title']], suffixes=('', '_fr'))
u = cat_merge(u, cat['gb'][['id', 'title']], suffixes=('', '_gb'))
u.rename(columns={'title': 'title_de'}, inplace=True)
u
uid_lists = {code: set(dat[code]['video_id']) for code in country_list}
style.use('default')
comp = np.array([(len(c0[1].intersection(c1[1])) if c0 != c1 else 0)
                 for c0, c1 in product(uid_lists.items(), repeat=2)]).reshape((5, 5))
plt.imshow(comp);
plt.colorbar();
plt.xticks(range(5), country_list);
plt.yticks(range(5), country_list);
plt.title('pairwise overlap of trending videos by country');
plt.show()
print("unique trending videos per country:")
print({n: len(l) for n, l in uid_lists.items()})
plt.imshow(comp / np.array([len(c) for c in uid_lists.values()]).reshape((-1, 1)));
plt.colorbar();
plt.xticks(range(5), country_list);
plt.yticks(range(5), country_list);
plt.title('pairwise overlap % of trending videos by country (% of left column / y axis)');
plt.show()
cat_uid_lists = {(code, cat): set(dat[code][dat[code]['category_id'] == cat]['video_id']) 
                 for code in country_list 
                 for cat in list(u.id.unique().astype(np.int))}
icat2cat = dict(enumerate(list(u.id.unique().astype(np.int))))
cat2icat = {v: k for k, v in icat2cat.items()}
cat_ucsi = np.zeros((5, 5, 32))
cat_mci = np.zeros((5, 5, 32))
for ic0 in range(5):
    for ic1 in range(5):
        if ic0 != ic1:
            for icat in range(32):
                c0 = country_list[ic0]
                c1 = country_list[ic1]
                cat = icat2cat[icat]
                set0 = cat_uid_lists[(c0, cat)]
                set1 = cat_uid_lists[(c1, cat)]
                if len(set1) != 0:
                    cat_ucsi[ic0, ic1, icat] = len(set0.intersection(set1)) / len(set1)
                    cat_mci[ic0, ic1, icat] = len(set0.intersection(set1)) / len(set0.union(set1))
fig, ax = plt.subplots(8, 4, figsize=(16, 32))
plt.setp(ax, xticks=range(5), xticklabels=country_list, yticks=range(5), yticklabels=country_list)
icat = 0
for r in range(8):
    for c in range(4):
        cur_ax = ax[r, c]
        cat_name = u[u['id'] == str(icat2cat[icat])]['title_us'].iloc[0]
        cur_ax.set_title(cat_name)
        cur_ax.imshow(cat_ucsi[:, :, icat])
        icat += 1
plt.show()
acti = {code: dat[code][dat[code]['category_id'] == 29] for code in country_list}
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

bri = acti['gb']['title'].unique()
for code in country_list:
    print(color.BLUE + "Unique Nonprofits & Activism videos in [{0}]:\n".format(code) + color.END)
    for title in acti[code]['title'].unique():
        if title in bri:
            print(color.BOLD + color.RED + title + color.END)
        else:
            print(title)
    print("-" * 20)
music = {code: dat[code][dat[code]['category_id'] == 10] for code in country_list}
for code in country_list:
    print("{0}: {1}".format(code, len(music[code])))
for c0, c1 in combinations(country_list, r=2):
    titles = set(music[c0]['title']).intersection(set(music[c1]['title']))
    n = len(titles)
    print("Overlapping videos in music between [{0}] and [{1}]: {2}".format(c0, c1, n))
for title in list(set(music['ca']['title']).intersection(set(music['de']['title'])))[:30]:
    print(title)
print("...")
de_us = set(music['de']['title']).intersection(set(music['us']['title']))
de_gb = set(music['de']['title']).intersection(set(music['gb']['title']))
for title in set(music['ca']['title']).intersection(set(music['de']['title'])):
    if title in de_us.intersection(de_gb):
        print(color.GREEN + title + color.END)
    elif title in de_us:
        print(color.BLUE + title + color.END)
    elif title in de_gb:
        print(color.RED + title + color.END)
    else:
        print(title)
