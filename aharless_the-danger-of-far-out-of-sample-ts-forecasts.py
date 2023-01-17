import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
ets_forecasts = pd.read_csv('../input/ensemble-grocery-01/sub_ets_log.csv', index_col=0)

test = pd.read_csv('../input/favorita-grocery-sales-forecasting/test.csv', 

                   parse_dates=['date'], index_col=0)
ets_forecasts.head()
test.head()
results = test[['date']].join(ets_forecasts).sort_values('unit_sales',ascending=False)
results.head(20)