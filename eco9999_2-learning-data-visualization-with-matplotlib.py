# Importing field
# This section same with 'Chapter 1' data adjusting section.
# With some short cuts ;)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Reading...
data_set = pd.read_csv('../input/voice.csv')

# Converting... ( String to boolean)
data_set['label'] = [0 if each == 'male' else 1 for each in data_set['label']]

# Normalizing...
data_set = (data_set - np.min(data_set)) / (np.max(data_set) - np.min(data_set))

# Separeting...
label = data_set['label'].values

# Dropping...
data_set = data_set.drop(columns=['label'])

data_set.head()

# First column is in 0th index named 'meanfreq'
# Making an int index holder
col_index = 0

# Columns name
col_name_0 = data_set.columns[col_index]

# Columns values
col_vals_0 = data_set[col_name_0].values

print('columns name : ',col_name_0)
print('columns values : ',col_vals_0)
# plotting with lineplot.
plt.plot(col_vals_0, c='b')

plt.show()
# 'marker' keyword argument for chosing mark shape 'o' is point, '^' is triagle shape, '+' is plus sign , '*' is star... etc. also '1','2','3' is diffrent shapes.
# 'linestyle' keyword argument for lines between points but we don't want that we are trying to avoid continius like look. also it can be '-','--','-.',':' for another styles.
# 'color' keyword argument for color.( how shocking )
# 'markersize' keyword argument is markers size.
plt.plot(col_vals_0, marker='o', linestyle='None', color='blue', markersize=1)

plt.show()
# There is a shortcut:
# 'bo' means blue point. first letter is for color it's optional, second is for marker.
# ms is markersize initials.
plt.plot(col_vals_0, 'bo',ms=1)

plt.show()
# it is just a list of column values size.
x = list(range(len(col_vals_0)))

# y is sample values.
y = col_vals_0

# we are using 's' instate of 'ms' or 'markersize' in this function. It can be a float or an array of floats.
# x and y is axes.
# c is color
# also we can use alpha if need some transparence.
plt.scatter(x,y, s=1, c='b')

plt.show()
# it is just a list of column values size.
x = list(range(len(col_vals_0)))

# y is sample values.
y = col_vals_0

# creating a hexadecimal  color.
color_hex = '#ABCDEF'

plt.scatter(x,y, s=1, c=color_hex)

plt.show()
import random
# Random hex number.
# 0xffffff is upper limit for hex color.
random_int = random.randint(0, int('0xffffff', 0))

# Printing random integer number.
print('random_int:', random_int)

# '%' is like a convertion operator.
# But a hex color need to be 6 digits, to ensure this we have to add '0.6' statement in the middle of the '%' and 'X',
# The '0.6' statement is going to add zero to the left of the number until it has six digits.
# 'X' is indicates hexadecimal number.
int_to_hex = '%0.6X' % random_int

# Printing hexadecimal number.
print('int_to_hex:', int_to_hex)

# Adding a hash (#) for color format.
random_color = str('#' + str(int_to_hex))

# Printing random hexadecimal number with color format  
print('random_color:', random_color)

# Or we can simply use this!. its the same thing.
# random_color = str("#%0.6X" % random.randint(0, int('0xffffff', 0)))


x = list(range(len(col_vals_0)))
y = col_vals_0

plt.scatter(x,y, s=1, c=random_color)
plt.show()
# Drawing plot.
plt.plot(col_vals_0, 'bo',ms=1)

# Making simple annotate text.
plt.annotate(xy=(1500,0.05), s='Hello')

plt.show()
# Drawing plot.
plt.plot(col_vals_0, 'bo',ms=1)

# Sample to annotate.
sample = 666

# X coord. is sample number.
x = sample

# Y coord. is samples values.
y = col_vals_0[sample]

# Text string to annotate.
text = str(str(sample) + '. value')

# text coord. there is no special meaning of this numbers just this area looks more clear that the others i think.
textcoord = (1500,0.05)

# xy is annotates coordinate.
# s is string value
# xytext is texts coordinates.
# arrowpros is a dictionary of arrow properties.
# If you want to see full descriptions you can use print(help(plt.annotate)). actually you can use this 'help()' function for all 'plt' functions. it is very usefull!!
# also you can use 'arrowprops={'arrowstyle': 'simple'}' for a regular arrow but i like this wegde thing much more. 
plt.annotate(xy=(x,y), s=text, xytext=textcoord, arrowprops={'arrowstyle': 'wedge'})

plt.show()
# Drawing plot.
plt.plot(col_vals_0, 'bo',ms=1)

# Finding max value and it's index
max_value = max(col_vals_0)
max_value_index= list(col_vals_0).index(max_value)

# Finding min value and it's index
min_value = min(col_vals_0)
min_value_index= list(col_vals_0).index(min_value)


# Text strings to annotate.
max_text = 'max. value'
min_text = 'min. value'

# text coords. there is no special meaning of this numbers just this area looks more clear that the others i think.
max_textcoord = (2500,0.10)
min_textcoord = (1000,0.10)

# Drawing max. values annotate.
plt.annotate(xy=(max_value_index,max_value), s=max_text, xytext=max_textcoord, arrowprops={'arrowstyle': 'wedge'})

# Drawing min. values annotate.
plt.annotate(xy=(min_value_index,min_value), s=min_text, xytext=min_textcoord, arrowprops={'arrowstyle': 'wedge'})


plt.show()
# Drawing plot.
plt.plot(col_vals_0, 'bo',ms=1)

# Horizontal lines y coordinate.
hline_coord = 0.1

# Draws a horizontal line.
plt.axhline(hline_coord, c='m')

# Vertical lines x coordinate.
vline_coord = 2000

# Draws a vertical line.
plt.axvline(vline_coord, c='m')

plt.show()
# Drawing plot.
plt.plot(col_vals_0, 'bo',ms=1)

# setting limits.
x_first_value = 2000
x_second_value = 2500

# Draws a vertical span between values
# alpha for transperence
plt.axvspan(x_first_value,x_second_value, alpha=0.1, color='m')

# setting limits.
y_first_value = 0.1
y_second_value = 0.2

# Draws a horizontal span between values
# alpha for transperence
plt.axhspan(y_first_value,y_second_value, alpha=0.1, color='g')

plt.show()
# Drawing plot.
plt.plot(col_vals_0, 'bo',ms=1)

# polygones coordinates.
polygones_x_coords = [0,2000,2000,0]
polygones_y_coords = [0.10,0.10,0.25,0.25]

# fill function for drawing polygone.
plt.fill(polygones_x_coords, polygones_y_coords, c='r', alpha=0.1) 

plt.show()
# notch is a notch in mean of this data.
# vert means vertical visualization.
# patch_artist for coloring box.
plt.boxplot(col_vals_0, notch=True, vert=False, patch_artist=True)

plt.show()
# for adding diffrent color.
box = plt.boxplot(col_vals_0, notch=True, vert=False, patch_artist=True)

box['boxes'][0].set_color('pink')

plt.show()
# For plotting whole dataframe, first we have to change data frame to np array.
data = np.array(data_set)
# Labels for y coordinates labels.
labels = data_set.columns.values
plt.boxplot(data, notch=True, vert=False, patch_artist=True, labels=labels)

plt.show()
# This data is making no sens with this plot but I am just using for curiosity.
# Lets limit the data.
data = col_vals_0[:50]

# We can use marker, markersize, color and much more keyword arguments in this function.
plt.polar(data, color='m', alpha=0.50)

plt.show()
# To visualize a pie of polarplot, we have to make a subplot fisrt.
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
c = ax.plot(data, c='b', alpha=0.75)

# starting angle.
ax.set_thetamin(135)

# ending angle.
ax.set_thetamax(45)

plt.show()
# Also we can offset origin.
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
c = ax.plot(data,c='pink', alpha=0.75)

# starting angle.
ax.set_thetamin(135)

# ending angle.
ax.set_thetamax(45)

# Adding title.
ax.set_title('Pice of Pie')

#Offseting origin.
ax.set_rorigin(-0.25)

plt.show()
# Violin plot for one feature.
# Points is number of points in side distrubition drawings.
# vert is boolean for vertical plot.
# widths is widths of violins.
# showextrema is boolean for extrema point.
# showmedians is boolean for median point.
plt.violinplot(col_vals_0, points=len(col_vals_0), vert=True, widths=0.1,showmeans=True, showextrema=True, showmedians=True)

plt.show()
# For plotting whole dataframe, first we have to change data frame to np array.
data = np.array(data_set)

plt.violinplot(data)

plt.show()