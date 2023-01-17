import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('5_sales.csv')
# you can convert your data to Dataframe using the following code
#df = pd.DataFrame(data)
df.head(5)
df.groupby('SalesRep')
by_comp = df.groupby("SalesRep")
by_comp.mean()
df.groupby('SalesRep').mean()
by_comp.std()
by_comp.min()
by_comp.max()
by_comp.count()
by_comp.describe()
by_comp.describe().transpose()
by_comp.describe().transpose()['Amy']
