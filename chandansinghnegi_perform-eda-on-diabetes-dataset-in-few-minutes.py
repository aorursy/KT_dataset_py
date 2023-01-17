import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport
from sklearn.datasets import load_diabetes
diab_data=load_diabetes()

df=pd.DataFrame(data=diab_data.data,columns=diab_data.feature_names)
df.head()
df.columns
### To Create the Simple report quickly

profile = ProfileReport(df, title='Pandas Profiling Report')
profile.to_widgets()
profile.to_file("output.html")