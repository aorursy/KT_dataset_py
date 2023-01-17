import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')
df.head()
len(df) # number of samples
# number of categories. http://devdocs.io/pandas~0.22/generated/pandas.dataframe.nunique
df.nunique()
# number of samples per category.
# Found the cookbook at http://devdocs.io/pandas~0.22/tutorials
# http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%204%20-%20Find%20out%20on%20which%20weekday%20people%20bike%20the%20most%20with%20groupby%20and%20aggregate.ipynb#4.2-Adding-up-the-cyclists-by-weekday
# use 'count' aggregate function https://stackoverflow.com/a/19385591/630752
grouped = df.groupby('landmark_id').aggregate('count')
grouped # note how we now have the same number of rows as the number of categories
# visualize distribution. http://devdocs.io/pandas~0.22/generated/pandas.series.sort_values
sorted_counts = grouped['id'].sort_values(ascending=False)
sorted_counts
plt.hist(sorted_counts) # histogram
sorted_counts.mean() # mean of series. http://devdocs.io/pandas~0.22/generated/pandas.series.mean
sorted_counts.median() # median http://devdocs.io/pandas~0.22/generated/pandas.series.median
sorted_counts_no_outliers = [x for x in sorted_counts if x < 100] # remove outliers
sorted_counts_outliers = [x for x in sorted_counts if x >= 100] # outliers
# zoom in histogram. http://devdocs.io/matplotlib~2.1/_as_gen/matplotlib.pyplot.hist
plt.hist(sorted_counts_no_outliers)
# plot outliers starting from 100. http://devdocs.io/matplotlib~2.1/_as_gen/matplotlib.pyplot.hist
plt.hist(sorted_counts_outliers, range=(100,2000))