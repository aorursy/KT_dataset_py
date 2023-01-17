import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline
categories = pd.read_csv('../input/documents_categories.csv')

entities = pd.read_csv('../input/documents_entities.csv')

meta = pd.read_csv('../input/documents_meta.csv')

topics = pd.read_csv('../input/documents_topics.csv')

events = pd.read_csv('../input/events.csv')

page_views = pd.read_csv('../input/page_views_sample.csv')

promoted = pd.read_csv('../input/promoted_content.csv')
# Investigating categories dataset

categories.columns
print ("Total categories: {}".format(categories.category_id.unique().shape[0]))

print ("Unique documents (web pages): {}".format(len(set(categories.document_id))))
categories.confidence_level.hist(bins=100)
categories.confidence_level.describe()
# Investigating Entities
entities.columns
print ("Unique documents: {}".format(entities.document_id.unique().shape[0]))

print ("Unique entities - derived from documents: {}".format(entities.entity_id.unique().shape[0]))

entities.confidence_level.hist(bins=100)

plt.title("Confidence level of entity classification")

# Much better classification of entities than topics
# Meta

meta.head()
# Topics

topics.head()
topics.confidence_level.hist(bins=100)

plt.title("Confidence of document topic classification")
page_views.head()