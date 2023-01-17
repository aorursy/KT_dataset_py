import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
dataset = {'research': [15000, 90000, 250000, 175000, 88000, 210000],
           'marketing': [5000, 25000, 31000, 44000, 19700, 21111],
           'city': ['Texas', 'Delaware', 'Florida', 'Texas', 'Delaware','Florida'],
           'profit': [9999, 5555, 3333, 4444, 1111, 2222]}
                
df = pd.DataFrame(dataset)
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2])], remainder='passthrough')
df = np.array(columnTransformer.fit_transform(df), dtype = np.str)
print(df)