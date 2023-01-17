import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

UFO_data = pd.read_csv('../input/scrubbed.csv')
UFO_data.head()

UFO_data.describe()