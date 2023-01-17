import pandas as pd
import numpy as np

import matplotlib.pylab as pylab
pylab.style.use('ggplot')

train = pd.read_csv("../input/train.csv", quotechar='"')
print(train.head(2))
%matplotlib inline
counts = train.groupby(['Dates', 'X', 'Y']).size()

counts.value_counts().plot('bar', logy=True)
counts.value_counts()
other = pd.DataFrame(counts[counts>=13])
other = other.reset_index()
manyarrests = train.merge(other, how='right')
manyarrests[manyarrests[0]==16]
manyarrests[manyarrests[0]==14]