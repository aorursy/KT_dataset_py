import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('max_rows',200)
df = pd.read_excel('/kaggle/input/passports-visa-strength/visa totals.xlsx',index_col=[0])
df.sort_values(by='Visa Index',ascending=False).round(1)