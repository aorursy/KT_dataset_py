# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.a
from clustergrammer2 import net

df = {}
def drop_ribo_mito(df):

    all_genes = df.index.tolist()

    print(len(all_genes))

    keep_genes = [x for x in all_genes if 'RPL' not in x]

    keep_genes = [x for x in keep_genes if 'RPS' not in x]

    print(len(keep_genes))



    df = df.loc[keep_genes]

    df.shape



    # Removing Mitochondrial Genes

    list_mito_genes = ['MTRNR2L11', 'MTRF1', 'MTRNR2L12', 'MTRNR2L13', 'MTRF1L', 'MTRNR2L6', 'MTRNR2L7',

                    'MTRNR2L10', 'MTRNR2L8', 'MTRNR2L5', 'MTRNR2L1', 'MTRNR2L3', 'MTRNR2L4']



    all_genes = df.index.tolist()

    mito_genes = [x for x in all_genes if 'MT-' == x[:3] or 

                 x.split('_')[0] in list_mito_genes]

    print(mito_genes)



    keep_genes = [x for x in all_genes if x not in mito_genes]

    df = df.loc[keep_genes]

    

    return df
def umi_norm(df):

    # umi norm

    barcode_umi_sum = df.sum()

    df_umi = df.div(barcode_umi_sum)

    return df_umi
df['ini'] = pd.read_csv('../input/GSE100866_CD8_merged-RNA_umi.csv', index_col=0)

new_cols = [(x, 'CD8 Status: ' + x.split('_')[1]) for x in df['ini'].columns.tolist()]

print(df['ini'].shape)

df['ini'].columns = new_cols

df['ini'].head()
df['ash'] = np.arcsinh(df['ini']/5)

df['ash-umi'] = umi_norm(drop_ribo_mito(df['ash']))

df['ash-umi'].shape
ser_var = df['ash-umi'].var(axis=1).sort_values(ascending=False)

keep_genes = ser_var.index.tolist()[:1000]



df['ash-umi-var'] = df['ash-umi'].loc[keep_genes]

df['ash-umi-var'].shape
net.load_df(df['ash-umi-var'])

net.normalize(axis='row', norm_type='zscore')

net.clip(-5,5)

net.load_df(net.export_df().round(2))

net.widget()