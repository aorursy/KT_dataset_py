import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv('../input/archive.csv')

data.describe()