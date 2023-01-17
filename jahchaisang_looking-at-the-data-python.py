import pandas as pd

import numpy as np
doc_ad_df = pd.read_csv('../input/promoted_content.csv', nrows=5)

doc_ad_df.head()



# Here we have the information about each promoted content (ad). Its id, document, campaign id, and advertiser id.
doc_meta_df = pd.read_csv('../input/documents_meta.csv', nrows=5)

doc_meta_df.head()



# Here we have the document id, source id, publisher id, and the time the promoted contents are published.
doc_cat_df = pd.read_csv('../input/documents_categories.csv', nrows=5)

doc_cat_df.head()



# Here's the categories of the documents (aka. promoted contents). And Outbrain's confidence for category assignments. 
doc_en_df = pd.read_csv('../input/documents_entities.csv', nrows=5)

doc_en_df.head()



# Kaggle said "an entity_id can represent a person, organization, or location." So this is person, organization, location this document is referring to along with Kaggle's NER confidence.
doc_top_df = pd.read_csv('../input/documents_topics.csv', nrows=5)

doc_top_df.head()



# Okay, this is the topic of the article.
cli_df = pd.read_csv('../input/clicks_train.csv', nrows=5)

cli_df.head()



# They collect click label based on ad_id. This is what we are predicting.

# Display id is the set of recommendations during the click
events_df = pd.read_csv('../input/events.csv', nrows=5)

events_df.head()



# each event here is a click event? display_id is the set of recommendations shown during the click. 

# this table shows what promoted content was clicked, the time, platform, geolocation.
pv_df = pd.read_csv('../input/page_views_sample.csv', nrows=5)

pv_df.head()



# This is all the page views that Outbrain is able to track, I guess? uuid is practically unique id of the event. The time, platform, location, and traffic source of the page view.