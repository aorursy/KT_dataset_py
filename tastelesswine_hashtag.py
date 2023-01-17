import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from pandas import DataFrame

from pandas import Series

import matplotlib.pyplot as plt

data = pd.read_csv("../input/Top_hashtag.csv")

data.shape

x1=data['Hashtag']

y1=data['Posts']

l=data['Likes']

c=data['Comments']
df = pd.DataFrame(data)
import matplotlib.pyplot as plt

 

# x-coordinates of left sides of bars 

#da=data.top()

left = [1, 2, 3, 4, 5]

 

# heights of bars

height = [4967, 6833, 893, 813, 3473]

 

# labels for bars

tick_label = ['love', 'freind', 'beach' 'family', 'yellow']

 

# plotting a bar chart

plt.bar(left, height, tick_label = tick_label,

        width = 0.8, color = ['blue', 'blue'])

 

# naming the x-axis

plt.xlabel('x - axis')

# naming the y-axis

plt.ylabel('y - axis')

# plot title

plt.title('My bar chart!')

 

# function to show the plot

plt.show()