import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter



%matplotlib inline
!ls -lSh wage-estimates/*csv
!cat wage-estimates/wm.industry.csv
!cat   wage-estimates/wm.estimate.csv
!cat wage-estimates/wm.ownership.csv
!cat wage-estimates/wm.datatype.csv
!cat  wage-estimates/wm.footnote.csv 
!cat wage-estimates/wm.seasonal.csv
# Input csv is having extra column values for few rows, so `usecols` is here to help.

area_df = pd.read_csv('wage-estimates/wm.area.csv',

            usecols=['area_code', 'area_text', 'display_level', 'selectable', 'sort_sequence'])



# cleaning leading or trailing spaces

area_df.display_level = area_df.display_level.apply(lambda x: x.strip())
area_df.head()
area_df.info()
# both values are matching ==> they uniq values -> data is fine.

for col in area_df.columns:

    print(col, '\tColumn Uniq values', len(area_df[col].unique()), len(area_df[col]))
# Lets check those Area's which has multiple Area Codes

tmp = area_df.area_text.value_counts()

tmp = dict(tmp[tmp > 1])

area_df[area_df.area_text.map(lambda x: tmp.get(x, 0) > 1)]
tmp = area_df.display_level.value_counts()

tmp.head(5)
plt.figure(figsize=(15, 5))



sns.distplot(tmp[2:])
# let check the length of these names

tmp.index.map(len)
Counter(tmp.index.map(len))
print(sorted(tmp.index[tmp.index.map(lambda x: -1 < len(x) < 4) ]))
print(sorted(tmp.index[tmp.index.map(lambda x: 3 < len(x) < 6) ]))
tmp = sorted(tmp.index[tmp.index.map(lambda x: 6 < len(x)) ])

print(tmp)
Counter([each.split(' ', 1)[1] for each in tmp if ' ' in each])
_ = area_df.selectable.value_counts().plot('pie')
plt.figure(figsize=(15, 5))

tmp = area_df.sort_sequence.value_counts()



tmp.head()
tmp[1:].plot()
all_data_df = pd.read_csv('wage-estimates/wm.data.1.AllData.csv')

all_data_df.shape
all_data_df.head()
all_data_df.info()
# all series_id's are unique

all_data_df.series_id.unique().shape[0] == 255479
all_data_df.year.value_counts().plot(kind='pie')
all_data_df.period.value_counts().plot(kind='pie')
plt.figure(figsize=(15, 5))

sns.distplot(all_data_df.value)
!head -n2 wage-estimates/wm.series.csv
import csv

fp = csv.reader(open('wage-estimates/wm.series.csv'))



def re_format(data):

    if len(data) > 16:

        return data[:10] + [''.join(data[10:-5]).strip()] + data[-5:]

    return data



series_df = list(map(re_format, fp))

series_df = pd.DataFrame(series_df[1:], columns=series_df[0])
series_df.shape
series_df.info()
series_df.describe().T
series_df['series_title footnote_codes'.split()].head().values
series_df.series_title.unique().tolist()