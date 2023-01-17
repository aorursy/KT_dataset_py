%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
tr = pd.read_csv('../input/train.csv', parse_dates=['activation_date'])  # 1503424
te = pd.read_csv('../input/test.csv',  parse_dates=['activation_date'])  # 508438
daset = pd.concat([tr, te], axis=0)
punct = set(string.punctuation)
print(punct)
emoji = set()
for s in daset['title'].fillna('').astype(str):
    for c in s:
        if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
            continue
        emoji.add(c)

for s in daset['description'].fillna('').astype(str):
    for c in str(s):
        if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
            continue
        emoji.add(c)
        
print(''.join(emoji))
# basic word and char stats for title
daset['n_titl_len'] = daset['title'].fillna('').apply(len)
daset['n_titl_wds'] = daset['title'].fillna('').apply(lambda x: len(x.split(' ')))
daset['n_titl_dig'] = daset['title'].fillna('').apply(lambda x: sum(c.isdigit() for c in x))
daset['n_titl_cap'] = daset['title'].fillna('').apply(lambda x: sum(c.isupper() for c in x))
daset['n_titl_spa'] = daset['title'].fillna('').apply(lambda x: sum(c.isspace() for c in x))
daset['n_titl_pun'] = daset['title'].fillna('').apply(lambda x: sum(c in punct for c in x))
daset['n_titl_emo'] = daset['title'].fillna('').apply(lambda x: sum(c in emoji for c in x))

# some ratio stats for title
daset['r_titl_wds'] = daset['n_titl_wds']/(daset['n_titl_len']+1)
daset['r_titl_dig'] = daset['n_titl_dig']/(daset['n_titl_len']+1)
daset['r_titl_cap'] = daset['n_titl_cap']/(daset['n_titl_len']+1)
daset['r_titl_spa'] = daset['n_titl_spa']/(daset['n_titl_len']+1)
daset['r_titl_pun'] = daset['n_titl_pun']/(daset['n_titl_len']+1)
daset['r_titl_emo'] = daset['n_titl_emo']/(daset['n_titl_len']+1)

# basic word and char stats for description
daset['n_desc_len'] = daset['description'].fillna('').apply(len)
daset['n_desc_wds'] = daset['description'].fillna('').apply(lambda x: len(x.split(' ')))
daset['n_desc_dig'] = daset['description'].fillna('').apply(lambda x: sum(c in punct for c in x))
daset['n_desc_cap'] = daset['description'].fillna('').apply(lambda x: sum(c.isdigit() for c in x))
daset['n_desc_pun'] = daset['description'].fillna('').apply(lambda x: sum(c.isupper() for c in x))
daset['n_desc_spa'] = daset['description'].fillna('').apply(lambda x: sum(c.isspace() for c in x))
daset['n_desc_emo'] = daset['description'].fillna('').apply(lambda x: sum(c in emoji for c in x))
daset['n_desc_row'] = daset['description'].astype(str).apply(lambda x: x.count('/\n'))

# some ratio stats
daset['r_desc_wds'] = daset['n_desc_wds']/(daset['n_desc_len']+1)
daset['r_desc_dig'] = daset['n_desc_dig']/(daset['n_desc_len']+1)
daset['r_desc_cap'] = daset['n_desc_cap']/(daset['n_desc_len']+1)
daset['r_desc_spa'] = daset['n_desc_spa']/(daset['n_desc_len']+1)
daset['r_desc_pun'] = daset['n_desc_pun']/(daset['n_desc_len']+1)
daset['r_desc_row'] = daset['n_desc_row']/(daset['n_desc_len']+1)
daset['r_desc_emo'] = daset['n_desc_emo']/(daset['n_desc_len']+1)

daset['r_titl_des'] = daset['n_titl_len']/(daset['n_desc_len']+1)
text_feature = list(daset.filter(regex='r_titl|n_titl|r_desc|n_desc').columns)
data = daset.loc[~daset['deal_probability'].isnull(), text_feature+['deal_probability']]
data.head()
def plot_corr(corr,method):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 13))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('./corr_{}.jpg'.format(method))
corr = data.corr(method='pearson')
plot_corr(corr, 'pearson')
