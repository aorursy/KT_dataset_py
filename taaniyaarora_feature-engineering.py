import pandas as pd
import featuretools as ft
import os
# choosing event event000002387
df_particle = pd.read_csv("../input/train_1/event000002387-particles.csv") 
df_hits = pd.read_csv("../input/train_1/event000002387-hits.csv")
df_cells = pd.read_csv("../input/train_1/event000002387-cells.csv")
df_truth = pd.read_csv("../input/train_1/event000002387-truth.csv")
df_particle.shape
df_hits.info()
df_particle.info()
df_truth.info()
df_cells.info()
df_hits.head()
df_hits.tail(2)
df_particle.head()
df_particle.tail(2)
df_particle.describe()
df_truth.head()
df_truth.tail(2)
df_cells.head()
df_cells.tail()
# This file contains additional detector geometry information.

df_detectors = pd.read_csv("../input/detectors.csv")
# Each module has a different position and orientation described in the detectors file.

df_detectors
print(df_hits.shape)
df_hits.head(2)
print(df_particle.shape)
df_particle.head(2)
print(df_truth.shape)
df_truth.head(2)
df_hits.index
hits_truth = df_hits.set_index('hit_id').join(df_truth.set_index('hit_id'))
df_hits.head(1)
df_truth.head(1)
hits_truth.head()
hits_truth.reset_index(inplace=True)
hits_truth.head(2)
hits_truth.shape
df_particle.shape
df_particle.head(2)
## Creating Entity set

es = ft.EntitySet(id="trackml")
es1 = es.entity_from_dataframe(entity_id='hits_truth', dataframe=hits_truth,
                               index = 'hit_id',
                               variable_types = { "volume_id":ft.variable_types.Categorical,
                                                  "layer_id":ft.variable_types.Categorical,
                                                  "module_id":ft.variable_types.Categorical })
es2 = es1.entity_from_dataframe(entity_id='particle', dataframe=df_particle,
                               index = 'particle_id' )
es2
# Defining one-to-many relationships among features of different entities

relation1 = ft.Relationship(es2['particle']['particle_id'],es2['hits_truth']['particle_id'])
relation1
es2.add_relationships([relation1])
es2.entities
df_hits.head()
df_cells.head()
