import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('/kaggle/input/cusersmarildownloadsearningcsv/earning.csv',delimiter=';',index_col=[0])
df['ratioprofessionals'] = df['femaleprofessionals']/df['maleprofessionals']

df[['femaleprofessionals','maleprofessionals']]