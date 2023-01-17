import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
clicks = pd.read_csv('../input/clicks_train.csv')

doc_cat = pd.read_csv('../input/documents_categories.csv')

doc_ent = pd.read_csv('../input/documents_entities.csv')

doc_meta = pd.read_csv('../input/documents_meta.csv')

doc_topics = pd.read_csv('../input/documents_topics.csv')

events = pd.read_csv('../input/events.csv')

pageviews = pd.read_csv('../input/page_views_sample.csv')

promoted_content = pd.read_csv('../input/promoted_content.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
clicks.head()
doc_cat.head()
doc_ent.head()
doc_meta.head()
doc_topics.head()
events.head()