%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],

        'pre_score': [4, 24, 31, 2, 3],

        'mid_score': [25, 94, 57, 62, 70],

        'post_score': [5, 43, 23, 23, 51]}

df = pd.DataFrame(raw_data, columns = ['first_name', 'pre_score', 'mid_score', 'post_score'])

df
#Make plot

# Create a list of the mean scores for each variable

mean_values = [df['pre_score'].mean(), df['mid_score'].mean(), df['post_score'].mean()]

# Create a list of variances, which are set at .25 above and below the score

variance = [df['pre_score'].mean() * 0.25, df['pre_score'].mean() * 0.25, df['pre_score'].mean() * 0.25]

# Set the bar labels

bar_labels = ['Pre Score', 'Mid Score', 'Post Score']

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



plt.show()
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



Ages_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] # age range for the developers, X axis values.



dev_y = [38496, 42000, 46752, 49320, 53200,     

         56000, 62316, 64928, 67317, 68748, 73752] #salaries, y axis values.

plt.bar(Ages_x, dev_y, color = 'k', label = 'All devs')

#notice the additional args color and linestyle and marker

#py_dev_y = [45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000, 71496, 75370, 83640]



#plt.plot(Ages_x, py_dev_y,color = 'b',marker = 'o', label = 'Python') 



plt.xlabel('Ages')

plt.ylabel('Median Salary(USD)')

plt.title('Median Salary (USD) by Age')



plt.legend()