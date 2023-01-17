# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
os.listdir('../input')
documents_meta = pd.read_csv('../input/documents_meta.csv')

documents_meta.head(3)
documents_meta.shape
documents_meta.publisher_id.unique().size
documents_categories = pd.read_csv('../input/documents_categories.csv')

documents_categories.head(3)
documents_categories.category_id.unique().size
documents_entities = pd.read_csv('../input/documents_entities.csv')

documents_entities.head(3)
documents_topics = pd.read_csv('../input/documents_topics.csv')

documents_topics.head(3)
documents_topics.topic_id.unique().size
page_views_sample = pd.read_csv('../input/page_views_sample.csv')

page_views_sample.head(3)
import matplotlib.pyplot as plt

% matplotlib inline

page_views_documment = page_views_sample.groupby(['document_id']).uuid.count().sort_values()



page_views_documment.head(3)
# transform data

page_views_documment_log = [np.log(x) for x in page_views_documment.values]
plt.boxplot(page_views_documment_log )

plt.show()
plt.hist(page_views_documment_log, bins=20)



plt.xlabel('Log(counts)')



plt.show()