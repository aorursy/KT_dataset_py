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
df_click_test = pd.read_csv('../input/clicks_test.csv')
df_click_test.head()
df_click_test.shape
del df_click_test
df_click_train = pd.read_csv('../input/clicks_train.csv')
df_click_train.head()
df_click_train.shape
del df_click_train

df_documents_categories = pd.read_csv('../input/documents_categories.csv')
df_documents_categories.head()
df_documents_categories.shape
df_documents_entities=pd.read_csv('../input/documents_entities.csv')
df_documents_entities.head()
df_documents_meta = pd.read_csv('../input/documents_meta.csv')
df_documents_meta.head()
df_documents_meta.shape
df_documents_meta.describe()
del df_documents_categories

del df_documents_meta
df_documents_topics = pd.read_csv('../input/documents_topics.csv')
df_documents_topics.head()
df_documents_topics.shape
df_documents_topics.describe()
df_events = pd.read_csv('../input/events.csv')
df_events.head()
df_events.shape
df_events.describe()
del df_events

del df_documents_topics
df_page_views_sample = pd.read_csv('../input/page_views_sample.csv')
df_page_views_sample.head()
df_page_views_sample.shape
del df_page_views_sample
df_promoted_content = pd.read_csv('../input/promoted_content.csv')
df_promoted_content.head()
df_promoted_content.shape
df_sample_submission = pd.read_csv('../input/sample_submission.csv')
df_sample_submission.head()
df_sample_submission.shape