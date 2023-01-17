import pandas as pd
df = pd.read_csv('/kaggle/input/seattle/seattle-properties.csv')
df.transpose()
df.plot(x='Street Address', y='Monthly Mortgage', kind='barh')
