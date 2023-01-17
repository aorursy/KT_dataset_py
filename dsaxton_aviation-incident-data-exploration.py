import numpy as np

import pandas as pd

from IPython.display import display, Markdown



df = pd.read_csv("../input/AviationDataUP.csv")



df.head(10)
pd.DataFrame({'missing_counts': df.apply(lambda x: np.sum(x.isnull())),

             'cardinality': df.apply(lambda x: x.nunique()),

             'data_types': df.dtypes})
pd.DataFrame(df.groupby('Investigation.Type').size())
pd.DataFrame(df.groupby(['Weather.Condition', 'Investigation.Type']).size())