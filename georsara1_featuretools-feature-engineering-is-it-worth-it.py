#Import libraries and data
import pandas as pd
import numpy as np
import featuretools as ft
import time
import matplotlib.pyplot as plt
import seaborn as sns

buro = pd.read_csv('../input/bureau.csv')
#Only read of data set
buro = buro.iloc[:50000,:]

buro = buro.reset_index()
#Create featuretool entities
es = ft.EntitySet(id="buro")

es = es.entity_from_dataframe(entity_id="buro",
                              dataframe=buro,
                              index="index",
                              #time_index="transaction_time",
                              #variable_types={"SK_ID_CURR": ft.variable_types.Categorical},
                              # "EDUCATION": ft.variable_types.Categorical,
                              # "MARRIAGE": ft.variable_types.Categorical,
                              #  }
                              )

es = es.normalize_entity(base_entity_id="buro",
                         new_entity_id="SK_ID_CURRENT",
                         index="SK_ID_CURR",
                         #additional_variables=["DAYS_CREDIT"]
                         )

#Run 'Deep Feature Synthesis' and record times
chunk_size = []
time_sec=[]

for c in range(250,5250,250):
    start = time.time()

    chunk = c

    print('Creating new features...')
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="SK_ID_CURRENT",
                                          agg_primitives = ["mean"],
                                          max_depth=1,
                                          chunk_size=chunk)

    stop = time.time()

    #Print elapsed time
    time_elapsed = stop - start
    #print('Chunk size =', chunk_size, ', Time = ', time_elapsed)
    chunk_size.append(chunk)
    time_sec.append(time_elapsed)
#Plot results
results = pd.DataFrame({'Chunk Size': chunk_size, 'Time in seconds': time_sec})

plt.figure()
sns.barplot('Chunk Size','Time in seconds', data = results)
plt.xticks(rotation=90)
plt.show()
