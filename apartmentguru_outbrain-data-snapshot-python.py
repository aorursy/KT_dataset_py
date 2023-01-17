

import pandas as pd

import numpy as np

%matplotlib inline  
# Yeah, I could have made a for loop. Some men would rather see the world burn. 



#df = pd.read_csv('../input/documents_categories.csv') #['document_id, category_id, confidence_level']

#df1 = pd.read_csv('../input/clicks_test.csv') # ['Display_id, ad_id']

#df2 = pd.read_csv('../input/documents_meta.csv') #['document_id, source_id, publisher_id, publish_time']

#df3 = pd.read_csv('../input/documents_entities.csv') #['document_id, entity_id, confidence_level']

#df4 = pd.read_csv('../input/promoted_content.csv') #['ad_id, document_id, campaign_id, advertister_id']

#df5 = pd.read_csv('../input/sample_submission.csv') #['display_id, ad_id']

#df6 = pd.read_csv('../input/documents_topics.csv') #['document_id, topic_id, confidence_level']

#df7 = pd.read_csv('../input/clicks_train.csv') # ['display_id, ad_id, clicked']

#df8 = pd.read_csv('../input/events.csv')# ['Display_id, uuid, document_id, timestamp, platform, geo_location']

###df9 = pd.read_csv('../input/page_views.csv') 

#df10 = pd.read_csv('../input/page_views_sample.csv') #['uuid, document_id, timestamp, platform, geo_location, traffic_source' ]
df = pd.read_csv('../input/documents_categories.csv') #['document_id, category_id, confidence_level']

print (df.count()) # 5481475  (int64)

print (df.head(10))


df1 = pd.read_csv('../input/clicks_test.csv') # ['Display_id, ad_id']

print (df1.count()) # 32225162

print (df1.head(10))
df2 = pd.read_csv('../input/documents_meta.csv') 

#['document_id, source_id, publisher_id, publish_time']

print (df2.head())

print (df2.count()) # Each id has a differing amount of data as a whole 

#dunno why but it's prob insignificant.
df3 = pd.read_csv('../input/documents_entities.csv') #['document_id, entity_id, confidence_level']

df3.count() #5537552
df4 = pd.read_csv('../input/promoted_content.csv') #['ad_id, document_id, campaign_id, advertister_id']

df.count() #5481475 

# This is the same as documents_categories.csv
df5 = pd.read_csv('../input/sample_submission.csv') #['display_id, ad_id']

print (df5.count())

print (df5.head())

#df5[1:3]

#seems like the ad_id uses really big numbers, probably like a barcode for iding
df6 = pd.read_csv('../input/documents_topics.csv') #['document_id, topic_id, confidence_level']

df6.count() # 11325960

df6.hist()

df6.plot(x='confidence_level', y='document_id', kind='kde')

#hmmm that's very interst...

# Changed x to be confid_lvl shit got effed up a bit

# 
df7 = pd.read_csv('../input/clicks_train.csv') # ['display_id, ad_id, clicked']

df7.count() #87141731
df8 = pd.read_csv('../input/events.csv')# ['Display_id, uuid, document_id, timestamp, platform, geo_location']

df8.count()