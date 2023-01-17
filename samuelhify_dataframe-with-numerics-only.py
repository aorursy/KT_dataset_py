import pandas as pd

import numpy as np

from io import StringIO



data = """

id,name

1,A

2,B

3,C

tt,D

4,E

5,F

de,G

1,3

0.12,4.322

r,3

"""

# read in dataframe

df = pd.read_csv(StringIO(data))



#splits dataframe into series and transform series to numeric type

series=[pd.to_numeric(df[colname], errors='coerce') for colname in df.columns]



# concats series(columns) back to dataframe

df2=pd.concat(series, axis=1)



# remove nan values

df2[(~df2.applymap(np.isnan)).all(1)]