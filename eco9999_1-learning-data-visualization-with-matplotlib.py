# import field

# numpy for numerical calculations.
import numpy as np

# pandas for dataframe class.
import pandas as pd

# matplotlib for plotting data.
from matplotlib import pyplot as plt

# blaa, blaa same old same old
# Let's move on...
# Reading from data with read_csv funtion
# Csv means comma separated values;

data_set = pd.read_csv('../input/voice.csv')

# Looking afew rows for firt impression with head() funciton.
data_set.head()

# Also I can use info() function for data types and data lengths.
data_set.info()
def values(label):
    
    # creating a list.
    dif_values = []
    
    # adding first value to my type list for comparison.
    dif_values.append(label[0])

    # checking all values in label.
    for each in label:
        # if it's not in types list it means different.
        if each not in dif_values:
            # Add it!
            dif_values.append(each)
            
    return dif_values
# copying data_sets label column to an another array.
label = data_set['label'].values

# let's try my function:
print(values(label))
# 'label' is an array with string values in it.
# type_1 is string in label array.
# type_1_value is type_1's boolean value that you can chose
# type_1_value can only be 0 and 1
def to_boolean(label, type_1, type_1_value):
    
    # creating a copy of my label array
    label_bool = label.copy()
    
    # making a for loop to comparison values.
    for index in range(len(label_bool)):
        # if my type_1 is in label list...
        if label_bool[index] == type_1:
            
            # add my value in it.
            label_bool[index] = type_1_value
            
        else:
            # my value can only be 0 or 1 !
            # if my value is 0 than make other value to 1
            # so if is 1 than 1 - 1 = 0, other value is 0
            #    if is 0 than 1 - 0 = 1, other value is 1
            label_bool[index] = (1 - type_1_value)

    return label_bool
# I want to make 'male' values to 0
# So 'female' values will be 1
# Result of the function is storing in 'label_bool' variable.
# For comparison I am giving another name 'label_bool', not 'label' again
label_bool = to_boolean(label,'male',0)

# Boolean conversion of label
print('boolean label : ',label_bool)

# Original label
print('original label : ',label)

# firt lets make label_bool equal to label array for future operations.
label = label_bool

print(values(label))
# Droping 'label' column from data_set
data_set = data_set.drop(columns=['label'])

# Or we can simply say inplace=True keyword argument, without equaling itself. it's the same thing, like this:
# data_set.drop(columns=['label'], inplace=True)

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
# Lets put columns values in it
plt.plot(col_vals_0)

# Show plot
plt.show()
# Adding label, and color
plt.plot(col_vals_0, color='green', label=col_name_0)

# Color can repretantable with letter c
# And for colors we can use it's first letter, like:
# g = green, r = red, b=blue, c=cyan, m=magenta, y=yellow, b=black etc...
# It is the same thing
plt.plot(col_vals_0, c='g', label=col_name_0)

plt.show()
# plotting same data
plt.plot(col_vals_0, color='green', label=col_name_0)

# legend is a label frame
plt.legend()

plt.show()
# second column is in 1st index named 'sd'
# making an int index holder
col_index = 1

# columns name
col_name_1 = data_set.columns[col_index]

# columns values
col_vals_1 = data_set[col_name_1].values

print('columns name : ',col_name_1)
print('columns values : ',col_vals_1)
# plotting column 0 
plt.plot(col_vals_0, color='green', label=col_name_0)

# plotting column 1 
plt.plot(col_vals_1, color='blue', label=col_name_1)

# adding legend
plt.legend()

plt.show()
# plotting column 0 
plt.plot(col_vals_0, color='green', label=col_name_0)

# plotting column 1 
plt.plot(col_vals_1, color='blue', label=col_name_1)

# if i expand it we cant see the position.
plt.legend(loc='upper right', ncol=2)

plt.show()
# plotting column 0 
plt.plot(col_vals_0, color='green', label=col_name_0)

# plotting column 1 
plt.plot(col_vals_1, color='blue', label=col_name_1)

# expanding
plt.legend(mode='expand')

plt.show()
# plotting data
plt.plot(col_vals_0, color='green', label=col_name_0)

# writing title
plt.title(col_name_0 + ' & ' + col_name_1)

# x coordinates label its in below
plt.xlabel('sample')

# x coordinates label its in left side
plt.ylabel(col_name_0 + ' values')

# adding legend for labels
plt.legend()

plt.show()
# this section is same 
plt.plot(col_vals_0, color='green', label=col_name_0)
plt.plot(col_vals_1, color='blue', label=col_name_1)
plt.title(col_name_0 + '&' + col_name_1)
plt.xlabel('index')
plt.ylabel('values')
plt.legend()

# it limits view 'xlim' for limiting x coordinate ylim for limitting y coordinate
# for seeing first 10 samples limit must be 0 to 9
plt.xlim(0,9)

# limiting values between 0.05 and 0.20 for seen better
plt.ylim(0.05,0.20)

plt.show()
# alpha must between 1 and 0 , 1 is solid 0 is transparent
plt.plot(col_vals_0, color='green', label=col_name_0, alpha=0.5)

# this section is same 
plt.title(col_name_0 + ' & ' + col_name_1)
plt.xlabel('sample')
plt.ylabel(col_name_0 + ' values')
plt.legend()


plt.show()
# this section is same 
plt.plot(col_vals_0, color='green', label=col_name_0)
plt.plot(col_vals_1, color='blue', label=col_name_1)
plt.title(col_name_0 + ' & ' + col_name_1)
plt.xlabel('index')
plt.ylabel('values')
plt.legend()
plt.xlim(0,9)
plt.ylim(0.05,0.20)

# ':' means dotted
# '-' solid line
# '--' dashed
# '-.' dash dot
plt.grid(color='grey', linestyle=':', linewidth=0.5)

plt.show()
# ---------------------first sub plot----------------------------

# creating subplot onject named first_plot 211 means 2 rows 1 column 1st elemnt
first_plot = plt.subplot(211)

#plotting with same way we did
first_plot.plot(col_vals_0, color='green', label=col_name_0)

# instead of title() we have to use set_title()
first_plot.set_title(col_name_0)

# instead of xlabel() we have to use set_xlabel()
first_plot.set_xlabel('index')

# instead of ylabel() we have to use set_ylabel()
first_plot.set_ylabel(col_name_0)

# instead of xlim() we have to use set_xlim()
first_plot.set_xlim(1700,1800)

# instead of ylim() we have to use set_ylim()
first_plot.set_ylim(0,0.25)

# ---------------------second sub plot---------------------------

# creating subplot onject named second_plot 211 means 2 rows 1 column 2nd elemnt
second_plot = plt.subplot(212)

#plotting with same way we did
second_plot.plot(col_vals_1, color='blue', label=col_name_1)

# instead of title() we have to use set_title()
second_plot.set_title(col_name_1)

# instead of xlabel() we have to use set_xlabel()
second_plot.set_xlabel('index')

# instead of ylabel() we have to use set_ylabel()
second_plot.set_ylabel(col_name_1)

# instead of xlim() we have to use set_xlim()
second_plot.set_xlim(1700,1800)

# instead of ylim() we have to use set_ylim()
second_plot.set_ylim(0,0.25)

# ------------------------------------------------------------

# Lets look at it
plt.show()
# creating subplot onject named first_plot 211 means 1 rows 2 column 1st elemnt
first_plot = plt.subplot(121)
first_plot.plot(col_vals_0, color='green', label=col_name_0)
first_plot.set_title(col_name_0)
first_plot.set_xlabel('index')
first_plot.set_ylabel(col_name_0)
first_plot.set_xlim(1700,1800)
first_plot.set_ylim(0,0.25)

# ------------------------------------------------------------

# creating subplot onject named second_plot 211 means 1 rows 2 column 2nd elemnt
second_plot = plt.subplot(122)
second_plot.plot(col_vals_1, color='blue', label=col_name_1)
second_plot.set_title(col_name_1)
second_plot.set_xlabel('index')
second_plot.set_ylabel(col_name_1)
second_plot.set_xlim(1700,1800)
second_plot.set_ylim(0,0.25)

# Lets look at it
plt.show()