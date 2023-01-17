import pandas as pd
import featuretools as ft
import matplotlib.pyplot as plt
import seaborn as sns
# choosing event event000002387
df_particle = pd.read_csv("../input/train_1/event000002387-particles.csv") 
df_hits = pd.read_csv("../input/train_1/event000002387-hits.csv")
df_cells = pd.read_csv("../input/train_1/event000002387-cells.csv")
df_truth = pd.read_csv("../input/train_1/event000002387-truth.csv")
df_particle.shape
df_hits.isnull().sum()
df_hits.info()
df_particle.isnull().sum()
df_particle.info()
df_truth.isnull().sum()
df_truth.info()
df_cells.isnull().sum()
df_cells.info()
df_hits.head(7)
df_hits.describe()
df_particle.head()
df_particle.tail()
df_particle.describe()
df_truth.head()
df_truth.tail()
df_cells.head()
df_cells.tail()
# This file contains additional detector geometry information.

df_detectors = pd.read_csv("../input/detectors.csv")
# Each module has a different position and orientation described in the detectors file.

df_detectors.head(7)
df_hits.nunique()
df_hits.volume_id.unique()
df_hits.layer_id.unique()
df_hits.module_id.unique()
df_test_hits = pd.read_csv('../input/test/event000000008-hits.csv')
df_test_cells = pd.read_csv('../input/test/event000000008-cells.csv')
df_test_hits.info()
df_test_hits.head()
df_test_hits.tail()
df_test_cells.info()
df_test_cells.head()
df_test_cells.tail()
## Creating Entity set

es = ft.EntitySet(id="hits")
es1 = es.entity_from_dataframe(entity_id='hits', dataframe=df_hits,
                               index = 'hit_id',
                               variable_types = { "volume_id":ft.variable_types.Categorical,
                                                  "layer_id":ft.variable_types.Categorical,
                                                  "module_id":ft.variable_types.Categorical })
es1['hits'].variables
es2 = es1.entity_from_dataframe(entity_id='particle', dataframe=df_particle,
                               index = 'particle_id' )
es2['particle']
df_cells.info()
df_cells.reset_index(inplace=True)
df_cells.head()                                    # value column signifies the amount of charge deposited by the particle
df_cells.tail()
es3 = es2.entity_from_dataframe(entity_id='cells', dataframe=df_cells,index='index'  )
es4 = es3.entity_from_dataframe(entity_id='truth',dataframe=df_truth, index='hit_id')
df_detectors.reset_index(inplace=True)
es5 = es4.entity_from_dataframe(entity_id='detectors', dataframe=df_detectors, index='index')
es5
es5.entities
# Defining one-to-many relationships among features of different entities

relation1 = ft.Relationship(es5['hits']['hit_id'],es5['cells']['hit_id'])

relation2 = ft.Relationship(es5['particle']['particle_id'],es5['truth']['particle_id'])
es5
es5.add_relationships([relation1,relation2])
es5.entities
%time feature_matrix, features = ft.dfs(entityset=es5, target_entity='particle',agg_primitives=['min','max'],max_depth=2)
df_particle.head(1)
df_truth.head(1)
feature_matrix
features
df_hits.head(2)
df_particle.head(2)
df_truth.head(2)
# obtaining the number of times each particle was detected

df_truth.groupby('particle_id')['hit_id'].count()
temp = df_truth[df_truth['particle_id']==4503874505277440]
temp
temp.weight.sum()
temp.count()
# the above particle was sensed/detected at 12 different positions on the detector as observed in df_particle dataframe below.

df_particle[df_particle['particle_id']==4503874505277440]
hits_list = temp.hit_id.tolist()
df_hits.loc[hits_list]
