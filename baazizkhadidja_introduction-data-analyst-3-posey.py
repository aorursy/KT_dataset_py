import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)









import pandas as pd

accounts = pd.read_csv("../input/posey/accounts.csv")

orders = pd.read_csv("../input/posey/orders.csv")

region = pd.read_csv("../input/posey/region.csv")

sales_reps = pd.read_csv("../input/posey/sales_reps.csv")

web_events = pd.read_csv("../input/posey/web_events.csv")

accounts.head()

accounts[['name','id']].head()

orders.info()
missing_data = accounts.isnull()
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")