import pandas as pd
import featuretools as ft

from datetime import datetime

from featuretools.primitives import *
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8'
}
to_read = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
to_parse = ['click_time']
df = pd.read_csv('../input/train_sample.csv', usecols=to_read, dtype=dtypes, parse_dates=to_parse)
df['id'] = df.index
# Create an entity set, a collection of entities (tables) and their relationships
es = ft.EntitySet(id='clicks')

# Create an entity "clicks" based on pandas dataframe and add it to the entity set
es = es.entity_from_dataframe(
    entity_id='clicks',
    dataframe=df,
    index='id',
    time_index='click_time',
    variable_types={
        # We need to set proper types so that Featuretools won't treat them as numericals
        'ip': ft.variable_types.Categorical,
        'app': ft.variable_types.Categorical,
        'device': ft.variable_types.Categorical,
        'os': ft.variable_types.Categorical,
        'channel': ft.variable_types.Categorical,
        'is_attributed': ft.variable_types.Boolean,
    }
)

# We can create new enities based on information we have, e.g. for ips or apps. We “normalize” the entity and extract a new one, this automatically adds a relationship between them
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='ip', index='ip')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='app', index='app')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='device', index='device')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='channel', index='channel')
es = es.normalize_entity(base_entity_id='clicks', new_entity_id='os', index='os')

# How our entityset looks like:
es
# Run Deep Feature Synthesis for app as a target entity (features will be create for each app)
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='app'
)

# List of created features:
feature_defs
# The features values
feature_matrix.head()
# Create feature with your own primitives
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='app',
    trans_primitives=[Hour],
    agg_primitives=[PercentTrue, Mode]
)

# List of created features:
feature_defs
# Tell Featuretools to add time when entity was last seen 
es.add_last_time_indexes()
    
train_cutoff_time = datetime.datetime(2017, 11, 8, 17, 0)
train_training_window = ft.Timedelta("1 day")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_entity='app',
    cutoff_time=train_cutoff_time,
    training_window=train_training_window,
)

feature_matrix.head()