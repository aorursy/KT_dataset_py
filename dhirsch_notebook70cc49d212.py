# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pvs = pd.read_csv('../input/page_views_sample.csv')
pvs.shape
pvs.head()
pvs.document_id.value_counts(normalize=True).head(n=15)
len(pvs.document_id.unique()) / pvs.shape[0]
len(pvs.document_id.unique())
docs = pvs.document_id.value_counts(normalize=True).head(n=100)
docs.hist()
pvs.document_id.value_counts().tail(n=10)
vc = pvs.document_id.value_counts()

len(vc == 1)
len(vc[vc == 1])
len(vc[vc < 10])