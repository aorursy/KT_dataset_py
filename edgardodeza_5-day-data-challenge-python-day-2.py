import numpy as np    # linear algebra
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt    # data visualization
df = pd.read_csv('../input/cereal.csv')    # read in CSV file 
df    # Alternatively use print(df)
print("The size of the dataframe is: ", df.shape)    # print the size of the dataframe
print("i.e. we have {} rows and {} columns.".format(df.shape[0], df.shape[1]))
print("The columns describe the following: \n")
print(df.columns)    # print the column names

sugars_column = df['sugars']    # select the 'sugars' column

print(sugars_column)
sugars_histogram = plt.hist(sugars_column)    # create histogram

# set the title and the labels for the x- and y- axis
plt.title('Distribution of sugars in one serving of cereal')
plt.xlabel('Sugars in grams')
plt.ylabel('Count')
plt.show()
# Change the plot style
plt.style.use('fivethirtyeight')

#------------------------------

# set the edge color to black
sugars_histogram = plt.hist(sugars_column, edgecolor = "black", color = 'orange')    

# set the title and the labels for the x- and y- axis
plt.title('Distribution of sugars in one serving of cereal')
plt.xlabel('Sugars in grams')
plt.ylabel('Count')
plt.show()
plt.style.use('fivethirtyeight')

#------------------------------
# set normed to True
sugars_histogram = plt.hist(sugars_column, edgecolor = "black", color = 'red', normed = True)    

plt.title('Distribution of sugars in one serving of cereal')
plt.xlabel('Sugars in grams')
plt.ylabel('Probability density')
plt.show()
plt.style.use('fivethirtyeight')

#------------------------------

# Change the number of bins to 2
sugars_histogram = plt.hist(sugars_column, edgecolor = "black", color = 'green', normed = True, bins = 2)    

# set the title and the labels for the x- and y- axis
plt.title('Distribution of sugars in one serving of cereal')
plt.xlabel('Sugars in grams')
plt.ylabel('Probability density')
plt.show()
print(sugars_histogram)    # tuple of bin_heights and bin_delimiters

bin_heights = sugars_histogram[0]
bin_delimiters = sugars_histogram[1]

print("\nThe bin heights are: ", bin_heights)
print("The bin delimiters are: ", bin_delimiters)

area = (7 - (-1)) * 0.06168831 + (15 - 7) * 0.06331169
print("\nThe area under the curve is: ", area)
plt.style.use('fivethirtyeight')

#------------------------------

weights = np.ones_like(sugars_column)/float(len(sugars_column))

# passing the weights array and setting normed to False
sugars_histogram = plt.hist(sugars_column, weights=weights, edgecolor = "black", color = 'purple', normed = False, bins = 2)    

# set the title and the labels for the x- and y- axis
plt.title('Distribution of sugars in one serving of cereal')
plt.xlabel('Sugars in grams')
plt.ylabel('Probability mass')
plt.show()
print(sugars_histogram)

bin_heights = sugars_histogram[0]
print("\nThe bin heights are: ", bin_heights)
print("\nThe sum of the bin heights is: ", sum(bin_heights))