%matplotlib inline
#
# Preparing the data
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/train.csv')

# Add column containing day of week expressed in integer
dow = {
    'Monday':0,
    'Tuesday':1,
    'Wednesday':2,
    'Thursday':3,
    'Friday':4,
    'Saturday':5,
    'Sunday':6
}
df['DOW'] = df.DayOfWeek.map(dow)

# Add column containing time of day
df['Hour'] = pd.to_datetime(df.Dates).dt.hour

# Retrieve categories list
cats = pd.Series(df.Category.values.ravel()).unique()
cats.sort()
#
# First, take a look at the total of all categories
#

plt.figure(1,figsize=(6,4))
plt.hist2d(
    df.Hour.values,
    df.DOW.values,
    bins=[24,7],
    range=[[-0.5,23.5],[-0.5,6.5]]
)
plt.xticks(np.arange(0,24,6))
plt.xlabel('Time of Day')
plt.yticks(np.arange(0,7),['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.ylabel('Day of Week')
plt.gca().invert_yaxis()
plt.title('Occurance by Time and Day - All Categories')
#
# Now look into each category
#

plt.figure(2,figsize=(16,9))
plt.subplots_adjust(hspace=0.5)
for i in np.arange(1,cats.size + 1):
    ax = plt.subplot(5,8,i)
    ax.set_title(cats[i - 1],fontsize=10)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.hist2d(
        df[df.Category==cats[i - 1]].Hour.values,
        df[df.Category==cats[i - 1]].DOW.values, 
        bins=[24,7],
        range=[[-0.5,23.5],[-0.5,6.5]]
    )
    plt.gca().invert_yaxis()