import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
path = "../input/m5-forecasting-accuracy"
selling_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))

calendar = pd.read_csv(os.path.join(path, "calendar.csv"))

salestrain = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))
selling_prices.head()
sns.distplot(selling_prices['sell_price'],bins=int(180/5))

plt.xlim(0, 40)
calendar.head()
for i, var in enumerate(["year", "weekday", "month", "event_name_1", "event_name_2", 

                         "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]):

    plt.figure()

    g = sns.countplot(calendar[var])

    g.set_xticklabels(g.get_xticklabels(), rotation=90)

    g.set_title(var)
salestrain.head()
for i, var in enumerate(["state_id", "store_id", "cat_id", "dept_id"]):

    plt.figure()

    g = sns.countplot(salestrain[var])

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_title(var)