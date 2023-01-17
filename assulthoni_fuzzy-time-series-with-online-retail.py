import os
import sys
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pylab as plt

%pylab inline

!pip3 install -U git+https://github.com/PYFTS/pyFTS
from pyFTS.partitioners import Grid
from pyFTS.models import chen, cheng
from pyFTS.common import Util , Transformations
from pyFTS.benchmarks import Measures
raw_df = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')
raw_df.head()
raw_df.describe()
raw_df.describe(include=['O'])
raw_df.info()
raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
raw_df.info()
cancellation_dataset = raw_df.loc[raw_df['Invoice'].str.contains("C", regex=False, na=False)]
display(cancellation_dataset.sample(15))
idx_tmp = cancellation_dataset.index
raw_df = raw_df.drop(idx_tmp)
raw_df = raw_df.drop(raw_df.loc[raw_df.Quantity<0].index)
raw_df.shape
input_df = raw_df[['InvoiceDate', 'Quantity']]
input_df.head()
input_df = input_df.set_index('InvoiceDate')
input_df = input_df.groupby(pd.Grouper(freq='D')).sum()
input_df.head()
px.line(input_df, x=input_df.index, y="Quantity")
data = input_df.Quantity.values
tdiff = Transformations.Differential(1)

boxcox = Transformations.BoxCox(0)

# diff_data = tdiff.apply(data)
fs = Grid.GridPartitioner(data=data,npart=100)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[25,10])

fs.plot(ax)
model = cheng.TrendWeightedFTS(partitioner=fs)
model.fit(data)
# model.append_transformation(tdiff)
print(model)
Util.plot_rules(model, size=[25,10] , rules_by_axis=100)
prediction = model.predict(data)
fig, ax = plt.subplots(figsize=(20,10))

plot(data)
plot(prediction)
Measures.get_point_statistics(prediction, model)