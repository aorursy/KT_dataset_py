import os
print(os.listdir("../input/"))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/201801_Punctuality_Statistics_Full_Analysis.csv')

df_top = df.groupby("airline_name").filter(lambda x: len(x) > 50)

sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})

plt.figure(figsize=(25,15))
order_airline_cancel=df.groupby(["airline_name"])["number_flights_cancelled"].median().sort_values().index.values
a = sns.boxplot(x="airline_name", y="number_flights_cancelled", data=df,palette='rainbow', order=order_airline_cancel, linewidth=1, fliersize=1)
b = sns.stripplot(x="airline_name", y="number_flights_cancelled", data=df, jitter=True, order=order_airline_cancel, marker='o', alpha=0.5, color='black', size=1)
a.set_xlabel("X Label",fontsize=1)
a.set_xticklabels(a.get_xticklabels(), rotation=90)
plt.show()
plt.figure(figsize=(25,15))
order_airline_delay=df.groupby(["airline_name"])["average_delay_mins"].median().sort_values().index.values
b = sns.boxplot(x="airline_name", y="average_delay_mins", data=df,palette='rainbow', order=order_airline_delay, linewidth=1, fliersize=1)
b = sns.stripplot(x="airline_name", y="average_delay_mins", data=df, order=order_airline_delay, jitter=True, marker='o', alpha=0.5, color='black', size=1)
b.set_xlabel("X Label",fontsize=1)
b.set_xticklabels(b.get_xticklabels(), rotation=90)
plt.show()
plt.figure(figsize=(25,15))
order_airport=df.groupby(["reporting_airport"])["average_delay_mins"].median().sort_values().index.values
c = sns.boxplot(x=df["reporting_airport"], y="average_delay_mins", data=df,palette='rainbow', order=order_airport, linewidth=1, fliersize=1)
c = sns.stripplot(x=df["reporting_airport"], y="average_delay_mins", data=df,order=order_airport, jitter=True, marker='o', alpha=0.5, color='black', size=1)
c.set_xlabel("X Label",fontsize=1)
c.set_xticklabels(c.get_xticklabels(), rotation=90)
plt.show()