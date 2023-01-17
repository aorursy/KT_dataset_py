import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

# Input data files are available in the "../input/" directory.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#train = pd.read_csv('../input/train.csv')

#train_target = train['label']

#train_features = train.ix[:,'pixel0':]



clicks_train = pd.read_csv('../input/clicks_train.csv')

docs_categories = pd.read_csv('../input/documents_categories.csv')

docs_entities = pd.read_csv('../input/documents_entities.csv', nrows=5000)

docs_meta = pd.read_csv('../input/documents_meta.csv', nrows=5000)

docs_topics = pd.read_csv('../input/documents_topics.csv', nrows=5000)

events = pd.read_csv('../input/events.csv', nrows=5000)

page_views = pd.read_csv('../input/page_views_sample.csv', nrows=5000)

promoted_content = pd.read_csv('../input/promoted_content.csv')
print('CLICKS')

clicks_train.head()
print('DOC CATEGORY')

docs_categories.head()
print('DOC ENTITY')

docs_entities.head()
print('DOC META')

docs_meta.head()
print('DOC TOPIC')

docs_topics.head()
print('EVENT')

events.head()
print('PAGE VIEW')

page_views.head()
print('PROMOTED CONTENT')

promoted_content.head()
new = promoted_content.join(clicks_train, on='ad_id', rsuffix=2)

new = new.join(docs_categories, on='document_id', rsuffix=2)
new.head()
new.count