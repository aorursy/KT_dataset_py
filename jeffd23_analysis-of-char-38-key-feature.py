import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
from bokeh.charts import Histogram, output_notebook, show
from bokeh.models import NumeralTickFormatter
output_notebook()

act_train = pd.read_csv('../input/act_train.csv')
act_test = pd.read_csv('../input/act_test.csv')
people = pd.read_csv('../input/people.csv')

acts = act_train[['people_id', 'activity_id', 'outcome']]
peeps = people[['people_id', 'char_38']]
def string_to_int(series):
    series = series.fillna('type_0')
    series = series.apply(lambda x: x.split('_')[1])
    return series

acts.people_id = string_to_int(acts.people_id)
acts.activity_id = string_to_int(acts.activity_id)
peeps.people_id = string_to_int(peeps.people_id)

df = acts.merge(peeps, how='left', on='people_id')
df.head()
print("Total population distribution")
df.char_38.describe().round(2)

sample = df.sample(100000)
p = Histogram(df, values='char_38', color='orange', bins=12,
              title="Total Sample Distribution")
p.yaxis[0].formatter = NumeralTickFormatter(format="0")

show(p)
print("Failed 0 outcome")
df_fail = df[df.outcome == 0]
print(df_fail.char_38.describe().round(2))

sample = df_fail.sample(100000)
p = Histogram(df_fail, values='char_38', color='red', bins=12,
              title="Failed Outcome Distribution")
p.yaxis[0].formatter = NumeralTickFormatter(format="0")

show(p)
print("Successful 1 outcome")
df_success = df[df.outcome == 1]

print(df_success.char_38.describe().round(2))

sample = df_success.sample(100000) 
p = Histogram(df_success, values='char_38', color='green', bins=12,
              title="Successful Outcome Distribution")
p.yaxis[0].formatter = NumeralTickFormatter(format="0")

show(p)

sample = df.sample(100000) 
p = Histogram(sample, values='char_38', color='outcome',
              title="none",
              legend='top_right')

show(p)

df2 = df[df.char_38 > 0]
sample = df2.sample(100000) 
p = Histogram(sample, values='char_38', color='outcome',
              title="Histogram of char_38 for outcomes",
              legend='top_right')

show(p)
