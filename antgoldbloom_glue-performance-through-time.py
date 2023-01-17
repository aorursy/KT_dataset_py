import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/3. Technical Performance/Language/GLUE/GLUE performance in detail.xlsx',index_col=[0])

df.index = pd.to_datetime(df.index)

df['yearquarter'] = df.index.to_period("Q")

df.groupby('yearquarter')[['state of the art']].max()
df['human performance'].max()