import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
!ls -ltr ../input
inputfile = '../input/flavors_of_cacao.csv'
!head ../input/flavors_of_cacao.csv
df = pd.read_csv(inputfile)
df.head()
df.columns = ['name', 'bar_name', 'ref', 'year', 'coco_percent', 'location', 'rating', 'type', 'bean origin']
df.head()
df['coco_percent'] = df['coco_percent'].apply(lambda x: float(x.strip('%')))
df.head()
df.shape
df.describe()
df['name'].value_counts(dropna = False)
df['name'].value_counts(dropna = False)[:50].plot(kind = 'bar', figsize = (15, 10))
df['name'].value_counts(dropna = False)[-50:].plot(kind = 'bar', figsize = (15, 10))
df['bar_name'].value_counts(dropna = False)
df['bar_name'].value_counts(dropna = False)[:25].plot(kind = 'bar', figsize = (20, 10))
df['year'].value_counts(dropna = False).plot(kind = 'bar', figsize = (15, 10))