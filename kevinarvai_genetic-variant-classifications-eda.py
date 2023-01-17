import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style(style='whitegrid')

sns.set(font_scale=1.5);



import pandas as pd

import re
df = pd.read_csv('../input/clinvar_conflicting.csv', dtype={'CHROM': str, 38: str, 40: object})
df.CHROM.dtype
df.shape
df.groupby(['CHROM', 'POS', 'REF', 'ALT']).ngroups
df.CHROM.value_counts()
ax = sns.countplot(x="CLASS", data=df)

ax.set(xlabel='CLASS', ylabel='Number of Variants');
gene_ct = pd.crosstab(df.SYMBOL, df.CLASS, margins=True)
gene_ct = pd.crosstab(df.SYMBOL, df.CLASS, margins=True)

gene_ct.drop('All', axis=0, inplace=True)



# limit to the 50 most submitted genes for visualization

gene_ct = gene_ct.sort_values(by='All', ascending=False).head(50)

gene_ct.drop('All', axis=1, inplace=True)



gene_ct.plot.bar(stacked=True, figsize=(12, 4));
vt_ct = pd.crosstab(df.CLNVC, df.CLASS, margins=True)

vt_ct.drop('All', axis=0, inplace=True)



# limit to the 50 most submitted genes for visualization

vt_ct = vt_ct.sort_values(by='All', ascending=False)

vt_ct.drop('All', axis=1, inplace=True)



vt_ct.plot.bar(stacked=True, figsize=(12, 4));
df.EXON.fillna('0', inplace=True)

df['variant_exon'] = df.EXON.apply(lambda x: [int(s) for s in re.findall(r'\b\d+\b', x)][0])
exondf = pd.crosstab(df['variant_exon'], df['CLASS'])

exondf.plot.bar(stacked=True, figsize=(20, 5));

plt.xlim(-0.5, 20.5);
MC_list = df.MC.dropna().str.split(',').apply(lambda row: list((c.split('|')[1] for c in row)))

MC_encoded = pd.get_dummies(MC_list.apply(pd.Series).stack()).sum(level=0)

MC_encoded = MC_encoded.reindex(index=MC_list.index)



# Incorporate the transformed MC feature into the existing DataFrame

df = df.join(MC_encoded).drop(columns=['MC'])



# Transformed MC feature

MC_encoded.head()
mccounts= {0: {},

           1: {},

           'All': {}

          }



for col in MC_encoded.columns:

    for class_ in [0, 1]:

        mccounts[class_][col] = df.loc[df['CLASS'] == class_][col].sum()

    

    mccounts['All'][col] = df[col].sum()

    

mc_ct = pd.DataFrame.from_dict(mccounts)



mc_ct_all = mc_ct.sum(axis=0)

mc_ct_all.name = 'All'

mc_ct = mc_ct.append(mc_ct_all, ignore_index=False)
mc_ct.drop('All', axis=0, inplace=True)



mc_ct = mc_ct.sort_values(by='All', ascending=False)

mc_ct.drop('All', axis=1, inplace=True)



mc_ct.plot.bar(stacked=True, figsize=(12, 4));
sift_ct = pd.crosstab(df.SIFT, df.CLASS, margins=True)

sift_ct.drop('All', axis=0, inplace=True)



# limit to the 50 most submitted genes for visualization

sift_ct = sift_ct.sort_values(by='All', ascending=False)

sift_ct.drop('All', axis=1, inplace=True)



sift_ct.plot.bar(stacked=True, figsize=(12, 4));
pp_ct = pd.crosstab(df.PolyPhen, df.CLASS, margins=True)

pp_ct.drop('All', axis=0, inplace=True)



# limit to the 50 most submitted genes for visualization

pp_ct = pp_ct.sort_values(by='All', ascending=False)

pp_ct.drop('All', axis=1, inplace=True)



pp_ct.plot.bar(stacked=True, figsize=(12, 4));
df = pd.get_dummies(df, columns=['SIFT', 'PolyPhen'])
from itertools import combinations

from scipy.stats import chi2_contingency
# select a few categorical features

categoricals_index = pd.MultiIndex.from_tuples(combinations(['CHROM', 'REF', 'ALT', 'IMPACT', 'Consequence', 'SYMBOL', 'CLASS'], 2))

categoricals_corr = pd.DataFrame(categoricals_index, columns=['cols'])
def chisq_of_df_cols(row):

    c1, c2 = row[0], row[1]

    groupsizes = df.groupby([c1, c2]).size()

    ctsum = groupsizes.unstack(c1)

    # fillna(0) is necessary to remove any NAs which will cause exceptions

    return chi2_contingency(ctsum.fillna(0))[1]
categoricals_corr[ 'chi2_p'] =  categoricals_corr.cols.apply(chisq_of_df_cols)
categoricals_corr
categoricals_corr.index = categoricals_index

categoricals_corr = categoricals_corr.chi2_p.unstack()
categoricals_corr
corr = df.select_dtypes(exclude='object').corr()



import numpy as np

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12));



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True);



# Draw the heatmap with the mask and correct aspect ratio

g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5});





from matplotlib.patches import Rectangle



g.add_patch(Rectangle((1, 6), 3, 1, fill=False, edgecolor='blue', lw=4));
snvs = df.loc[(df.REF.str.len()==1) & (df.ALT.str.len()==1)]

indels = df.loc[(df.REF.str.len()>1) | (df.ALT.str.len()>1)]
len(df) == (len(snvs) + len(indels))
snp_indel = pd.concat([snvs.CLASS.value_counts(normalize=True).rename('snv_class'), 

                       indels.CLASS.value_counts(normalize=True).rename('indel_class')], 

                      axis=1).T
snp_indel.plot.bar(stacked=True, figsize=(12, 4));
clndn = pd.concat([df.CLASS.loc[(df.CLNDN=='not_specified') | (df.CLNDN=='not_provided') | (df.CLNDN=='not_specified|not_provided')].value_counts(normalize=True).rename('disease_not_specified'), 

                       df.CLASS.loc[(df.CLNDN!='not_specified') | (df.CLNDN!='not_provided') | (df.CLNDN!='not_specified|not_provided')].value_counts(normalize=True).rename('some_disease_specified')], 

                      axis=1).T
clndn.plot.bar(stacked=True, figsize=(12, 4));
sns.distplot(df.AF_ESP, label="AF_ESP")

sns.distplot(df.AF_EXAC, label="AF_EXAC")

sns.distplot(df.AF_TGP, label="AF_TGP")

plt.legend();