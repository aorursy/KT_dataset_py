# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import gc



# Any results you write to the current directory are saved as output.
click_test_df = pd.read_csv("../input/clicks_test.csv", chunksize = 10)

clicks_train_df = pd.read_csv("../input/clicks_train.csv", chunksize = 10)

documents_categories_df = pd.read_csv("../input/documents_categories.csv", chunksize = 10)

documents_entities_df = pd.read_csv("../input/documents_entities.csv", chunksize = 10)

documents_meta_df = pd.read_csv("../input/documents_meta.csv", chunksize = 10)

documents_topics_df = pd.read_csv("../input/documents_topics.csv", chunksize = 10)

events_df = pd.read_csv("../input/events.csv", chunksize = 10)

page_views_sample_df = pd.read_csv("../input/page_views_sample.csv", chunksize = 10)

promoted_content_df = pd.read_csv("../input/promoted_content.csv", chunksize = 10)

sample_submission_df = pd.read_csv("../input/sample_submission.csv", chunksize = 10)
print("clicks_train_df:\t",clicks_train_df.get_chunk().columns.tolist())

print("documents_categories_df:\t",documents_categories_df.get_chunk().columns.tolist())

print("documents_entities_df:\t",documents_entities_df.get_chunk().columns.tolist())

print("documents_meta_df:\t",documents_meta_df.get_chunk().columns.tolist())

print("documents_topics_df:\t",documents_topics_df.get_chunk().columns.tolist())

print("events_df:\t",events_df.get_chunk().columns.tolist())

print("page_views_sample_df:\t",page_views_sample_df.get_chunk().columns.tolist())

print("promoted_content_df:\t",promoted_content_df.get_chunk().columns.tolist())
#clicks_train_df = pd.read_csv("../input/clicks_train.csv",chunksize=10000000)

#promoted_content_df = pd.read_csv("../input/promoted_content.csv")

#train_promoted_df = pd.merge(clicks_train_df.get_chunk(), promoted_content_df,on = 'ad_id')

#del clicks_train_df

#gc.collect()

#train_promoted_df.info()
#print(train_promoted_df.columns.tolist())

#documents_categories_df = pd.read_csv("../input/documents_categories.csv")

#documents_topics_df = pd.read_csv("../input/documents_topics.csv")

#documents_entities_df = pd.read_csv("../input/documents_entities.csv")

#documents_meta_df = pd.read_csv("../input/documents_meta.csv")

events_df = pd.DataFrame()

for chunk in pd.read_csv("../input/events.csv",chunksize = 1000000):

    events_df = events_df.append(chunk)
#del documents_topics_df,documents_entities_df

#train_promoted_df

#gc.collect()

len(events_df.get_chunk())

#print(len(set(documents_categories_df.document_id)))



print(len(set(documents_categories_df.document_id).intersection(set(events_df.document_id))))