# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
documents_categories_df = pd.read_csv("../input/documents_categories.csv")

documents_entities_df = pd.read_csv("../input/documents_entities.csv")

documents_meta_df = pd.read_csv("../input/documents_meta.csv")

documents_topics_df = pd.read_csv("../input/documents_topics.csv")

events_df = pd.read_csv("../input/events.csv")

page_views_df = pd.read_csv("../input/page_views_sample.csv", nrows=50000)

promoted_content_df = pd.read_csv("../input/promoted_content.csv")

clicks_train_df = pd.read_csv("../input/clicks_train.csv", nrows=1000)



print("Dataframes count:")

print("documents_categories - {0}".format(len(documents_categories_df)))

print("documents_entities - {0}".format(len(documents_entities_df)))

print("documents_meta - {0}".format(len(documents_meta_df)))

print("documents_topics - {0}".format(len(documents_topics_df)))

print("events - {0}".format(len(events_df)))

print("page_views - {0}".format(len(page_views_df)))

print("promoted_content - {0}".format(len(promoted_content_df)))

print("clicks_train - {0}".format(len(clicks_train_df)))
clicks_train_df.head(10)
events_df.head(10)
promoted_content_df.head(10)
promoted_content_df.groupby(["ad_id"])["document_id"].count()
documents_meta_df.head(10)
documents_categories_df.head(10)
documents_entities_df.head(10)
documents_topics_df.head(10)
page_views_df.head(10)
page_views_df.groupby(['document_id'])['uuid'].count().sort_values(ascending=False)