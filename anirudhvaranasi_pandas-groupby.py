import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = { 'Company' : ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],

         'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'], 

         'Sales' : [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)

df
byComp = df.groupby('Company')

byComp.max()
df.groupby('Company').describe().transpose()