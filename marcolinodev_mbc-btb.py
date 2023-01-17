# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
doc_ad_df = pd.read_csv('../input/clicks_train.csv', nrows=5)

doc_ad_df.head()
doc_en_df = pd.read_csv('../input/documents_entities.csv', nrows=5)

doc_en_df.head()
doc_meta_df = pd.read_csv('../input/documents_meta.csv', nrows=5)

doc_meta_df.head()
doc_cat_df = pd.read_csv('../input/documents_categories.csv', nrows=5)

doc_cat_df.head()
xtrain = pd.read_csv('../input/clicks_train.csv')
xtrain = xtrain.ix[xtrain.clicked == 1]
xtrain.head()
freq_table = xtrain.ad_id.value_counts()

print(freq_table.head())
xtest = pd.read_csv('../input/clicks_test.csv')

xtest['count'] = xtest['ad_id'].map(freq_table)
xtest.sort_values(by=['display_id', 'count'], inplace = True, ascending = False) 
xsub = xtest.groupby('display_id').aggregate(lambda x: ' '.join([str(ff) for ff in x]))

xsub.to_csv('../input/mbc_sub01.csv', index = True)