%matplotlib inline



import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

import seaborn as sns; sns.set()



plt.rcParams['figure.figsize'] = (13, 16)
crimes = pd.read_csv('../input/Crime_Data_2010_2017.csv')
crimes.tail()
crimes['Date Occurred'] = pd.to_datetime(crimes['Date Occurred'], format="%m/%d/%Y")

crimes['Date Reported'] = pd.to_datetime(crimes['Date Reported'], format="%m/%d/%Y")
crimes['Time Occurred'] = crimes['Time Occurred'].astype(str).str.zfill(4)

crimes['Hour Occurred'] = crimes['Time Occurred'].apply(lambda t: int(t[:2]))
crimes['Delta of Report'] = (crimes['Date Reported'] - crimes['Date Occurred']).dt.days
crimes_from_15 = crimes[(crimes['Date Occurred'] >= '01/01/2015')]

print(crimes_from_15.shape)

crimes_from_15.tail()
gr_count = crimes_from_15.groupby(['Crime Code Description'], as_index=['Crime Code Description']).count().ix[:, 1]

gr_count
selected_crimes_from_15 = gr_count[gr_count > 20000]
selected_names = selected_crimes_from_15.index

print("\n".join(selected_names))
g = sns.FacetGrid(crimes_from_15, 

                  row="Crime Code Description", 

                  row_order=selected_names,

                  size=1.9, aspect=4, 

                  sharex=True,

                  sharey=False)



g.map(sns.distplot, "Hour Occurred", bins=24, kde=False, rug=False)
crimes_time_series = crimes_from_15.groupby(['Crime Code Description', 'Date Occurred'], as_index=['Crime Code Description', 'Date Occurred']).count().ix[:,1].unstack(level=0).unstack(level=0).fillna(0)
for i, col in zip(range(1, len(selected_names) + 1), selected_names):

    plt.subplot(len(selected_names), 1, i)

    plt.title(col)

    crimes_time_series[col].rolling(window=20, min_periods=20).mean().plot()
correlation_matrix = crimes_time_series.unstack(0)[selected_names].corr()



sns.heatmap(correlation_matrix)
deltas = crimes.groupby(['Crime Code Description'])['Delta of Report'].describe()

deltas[(deltas["50%"] >= 3) & (deltas["count"] > 50)]