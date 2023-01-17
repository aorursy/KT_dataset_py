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
# Read the data you need

df_train = pd.read_csv('../input/clicks_train.csv')

df_events = pd.read_csv('../input/events.csv')

df_promoted_content = pd.read_csv('../input/promoted_content.csv')

df_doc_categories = pd.read_csv('../input/documents_categories.csv')

df_doc_entities = pd.read_csv('../input/documents_entities.csv')

df_doc_meta = pd.read_csv('../input/documents_meta.csv')

df_doc_topics = pd.read_csv('../input/documents_topics.csv')

df_doc_categories = pd.read_csv('../input/documents_categories.csv')
page_views_events_joined_by_uuid_document_df = page_views_df.alias('page_views').join(events_df.alias('events'), on=['uuid','document_id'], how='outer')