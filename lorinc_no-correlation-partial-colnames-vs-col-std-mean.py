import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = 20, 20
src_test = pd.read_csv('../input/test.csv',index_col='id')

src_train = pd.read_csv('../input/train.csv',index_col='id')



src_test['target'] = -1

src = src_test.append(src_train, sort=None)



src_test = src_train = None
meta = (

    src.drop(['wheezy-copper-turtle-magic', 'target'], axis=1)

    .describe()

    .T

    .loc[:,['mean','std']])



meta['ix'] = meta.index

meta = meta.join(meta['ix'].str.split('-', expand=True))

meta = meta.join(meta[['ix', 0,1,2,3]].rank(), rsuffix='_rank')

meta = meta.join(meta[['ix', '0','1','2','3']].fillna('').applymap(len), rsuffix='_len')
corr = pd.DataFrame(

            meta.drop(

                ['ix', '0','1','2','3'], axis =1

            ), 

            columns = meta.drop(

                ['ix', '0','1','2','3'], axis =1

            ).columns).corr()



fig, ax = plt.subplots(figsize=(10, 10))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

plt.xticks(range(len(corr.columns)), corr.columns);

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()
