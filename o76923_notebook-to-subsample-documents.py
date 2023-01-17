from os import mkdir

from os.path import isdir

import shutil



if isdir('./events_100'):

    shutil.rmtree('./events_100')

mkdir('./events_100')
import pandas as pd

from numpy.random import choice



with open('../input/events.csv') as in_file:

    df = pd.read_csv(in_file,

                     header=0,

                     index_col=False,

                     usecols=('display_id', 'document_id', 'platform', 'geo_location', 'timestamp'),

                     dtype={'display_id': int, 'document_id': int, 'platform': str, 'geo_location': str, 'timestamp': int}

                    )

docs = choice(df['document_id'].unique(),100,False)

df = df[df['document_id'].isin(docs)]

x = df['geo_location'].apply(lambda x: (x.split('>') + [None, ]*3)[:3])

df['geo_0'] = x.apply(lambda x:x[0])

df['geo_1'] = x.apply(lambda x:x[1])

df['geo_2'] = x.apply(lambda x:x[2])

df.drop('geo_location', axis=1, inplace=True)

df['platform'] = df['platform'].astype('category')

df['timestamp'] = pd.to_datetime(df['timestamp']+1465876799998, unit='ms')

df.to_csv('./events_100/events.csv')
with open('./events_100/events.csv') as in_file:

    df = pd.read_csv(in_file,

                     header=0,

                     index_col=False,

                     usecols=('display_id', 'document_id'),

                     dtype={'display_id': int, 'document_id': int}

                    )

document_ids = df['document_id'].unique()

display_ids = df['display_id'].unique()

df = None
files = [

    {'filename': 'documents_categories', 'join_on': 'document_id', 'columns': {'category_id': int, 'confidence_level': float}},

    {'filename': 'documents_topics', 'join_on': 'document_id', 'columns': {'topic_id': int, 'confidence_level': float}},

    {'filename': 'documents_entities', 'join_on': 'document_id', 'columns': {'entity_id': str, 'confidence': float}},

    {'filename': 'documents_meta', 'join_on': 'document_id', 'columns': {'source_id': float, 'publisher_id': float, 'publish_time': str}},

    {'filename': 'promoted_content', 'join_on': 'document_id', 'columns': {'campaign_id': int, 'advertiser_id': int, 'ad_id': int}},

    {'filename': 'clicks_train', 'join_on': 'display_id', 'columns': {'ad_id': int, 'clicked': bool}},

]
for f in files:

    df = pd.read_csv("../input/%s.csv" % f['filename'],

                     header=0,

                     index_col=False,

                     usecols=list(f['columns'].keys()).extend(f['join_on']),

                     dtype=f['columns']

                    )

    if f['join_on'] == 'document_id':

        df = df[df['document_id'].isin(document_ids)]

    else:

        df = df[df['display_id'].isin(display_ids)]

    df.to_csv("./events_100/%s.csv" % f['filename'])

    df = None

    print("File %s has been created" % f['filename'])