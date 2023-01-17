import re



import numpy as np

import pandas as pd
data = pd.read_csv('../input/train_1.csv').iloc[:256]



data.head()
date_columns = [c for c in data.columns if re.match(r'\d{4}-\d{2}-\d{2}', c)]



print(date_columns[:5])

print(date_columns[-5:])
LAG_DAYS = 7
used_data = data[['Page'] + date_columns[LAG_DAYS:]]
flattened = pd.melt(used_data, id_vars='Page', var_name='date', value_name='Visits')

flattened.dropna(how='any', inplace=True)



flattened.head()
date_indices = {d: i for i, d in enumerate(date_columns)}
# We will need the page indices to tell us which row to look at

data['page_indices'] = data.index

# We set the index to page so we can merge with `flattened` easily

data.set_index('Page', inplace=True)



flattened['date_indices'] = flattened['date'].apply(date_indices.get)

flattened = flattened.set_index('Page').join(data['page_indices']).reset_index()



flattened.iloc[538:548] # 543 happens to be the index where the second time series begins
for lag in range(1, LAG_DAYS + 1):

    flattened['lag_%d' % lag] = data[date_columns].values[

        flattened['page_indices'],

        flattened['date_indices'] - lag

    ]
flattened.dropna(how='any', inplace=True)
flattened.shape
flattened.head()
flattened.iloc[543:548]
data.iloc[:2]
flattened.drop(['page_indices', 'date_indices'], inplace=True, axis=1)



flattened.head()