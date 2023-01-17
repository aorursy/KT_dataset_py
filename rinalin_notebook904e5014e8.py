# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cate = pd.read_csv('../input/documents_categories.csv')

cate.rename(columns = {'confidence_level':'cl_category'}, inplace = True)

cate=cate.iloc[cate.groupby('document_id')['cl_category'].agg(pd.Series.idxmax)]
cate[0:5]
entity = pd.read_csv('../input/documents_entities.csv')

entity.rename(columns = {'confidence_level':'cl_entity'}, inplace = True)

entity=entity.iloc[entity.groupby('document_id')['cl_entity'].agg(pd.Series.idxmax)]
entity[0:5]
meta = pd.read_csv("../input/documents_meta.csv")
meta[0:5]
topics = pd.read_csv('../input/documents_topics.csv')

topics.rename(columns = {'confidence_level':'cl_topic'}, inplace = True)

topics=topics.iloc[topics.groupby('document_id')['cl_topic'].agg(pd.Series.idxmax)]
topics[0:5]
documents = [entity,meta,topics]

base =cate

for doc in documents:

    base = base.merge(doc,how='outer',left_on='document_id',right_on='document_id')
base[0:5]