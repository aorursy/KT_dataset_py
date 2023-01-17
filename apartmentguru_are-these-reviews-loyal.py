%matplotlib inline
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/Reviews.csv')
df.head()
df['Score'].describe()

df['HelpfulnessDenominator'].describe()
df['HelpfulnessNumerator'].describe()
# Create a list of the mean scores for each variable
mean_values = [df['Score'].mean(), df['HelpfulnessDenominator'].mean(), df['HelpfulnessNumerator'].mean()]

# Create a list of variances, which are set at .25 above and below the score
variance = [df['Score'].mean() * 0.25, df['Score'].mean() * 0.25, df['Score'].mean() * 0.25]

# Set the bar labels
bar_labels = ['Rating', 'HelpDen', 'HelpNum']

# Create the x position of the bars
x_pos = list(range(len(bar_labels)))

# Create the plot bars
# In x position
plt.bar(x_pos,
        # using the data from the mean_values
        mean_values,
        # with a y-error lines set at variance
        yerr=variance,
        # aligned in the center
        align='center',
        # with color
        color='#FFC222',
        # alpha 0.5
        alpha=0.5)

# add a grid
plt.grid()

# set height of the y-axis
max_y = max(zip(mean_values, variance)) # returns a tuple, here: (3, 5)
plt.ylim([0, (max_y[0] + max_y[1]) * 1.1])

# set axes labels and title
plt.ylabel('Score')
plt.xticks(x_pos, bar_labels)
plt.title('Mean Scores For Each Test')

df['ProductId'].describe()
