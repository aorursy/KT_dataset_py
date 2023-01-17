import pandas as pd

import seaborn as sb
df = sb.load_dataset('tips')
df.head()
df.info()
df['per'] = df.tip / df.total_bill
df.head()