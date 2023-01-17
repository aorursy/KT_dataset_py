



import numpy as np 

import pandas as pd 

from subprocess import check_output
# Data files available.



print(check_output(["ls", "../input"]).decode("utf8"))
# Function to load data



def load(filestub):

    return pd.read_csv("../input/" + filestub + ".csv")



# clicks_train = pd.read_csv("../input/clicks_train.csv")

# clicks_test = pd.read_csv("../input/clicks_test.csv")

# documents_categories = pd.read_csv("../input/documents_categories.csv")

# documents_entities = pd.read_csv("../input/documents_entities.csv")

# documents_meta = pd.read_csv("../input/documents_meta.csv")

# documents_topics = pd.read_csv("../input/documents_topics.csv")

# events = pd.read_csv("../input/events.csv")

# page_views_sample = pd.read_csv("../input/page_views_sample.csv")

# promoted_content = pd.read_csv("../input/promoted_content.csv")

# sample_submission = pd.read_csv("../input/sample_submission.csv")





                         
# Get a look at it.



# data = [clicks_train, 

#         clicks_test, 

#         documents_categories, 

#         documents_entities, 

#         documents_meta,

#         documents_topics, 

#         events,

#         page_views_sample,

#         promoted_content, 

#         sample_submission]

# names_of_data = ["clicks_train", 

#         "clicks_test", 

#         "documents_categories", 

#         "documents_entities", 

#         "documents_meta",

#         "documents_topics", 

#         "events",

#         "page_views_sample",

#         "promoted_content", 

#         "sample_submission"]



# for d, s in zip(data, names_of_data):

#     print (s, d.shape)

#     print(d.head())

#     print()

    

    
#How many unique users in events data? 



print(events.shape)

ids = events['uuid']

a = ids.unique()

print(a[:5])

#How many unique users in page_views_sample?

#How many unique users in page_views?