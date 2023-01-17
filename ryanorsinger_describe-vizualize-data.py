# Run this block of code in order to import all of the libraries that we need for this module.

# Shift + Enter on your keyboard or click the play button to run code cells. 

import numpy as np # linear algebra library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Vizualization libraries

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
# Acquire the data 

quartet = pd.read_csv("../input/quartet.csv", index_col="id") # this is how the pandas library reads CSV files. Then we assign to a variable.
# There are 4 groups that each have an X value and a Y value

print(quartet)
# Let's look at the entire dataset

quartet.describe()
# Let's look at the average of each quartet and the standard deviation 

# Mean means the average value

# Standard deviation represents the typical distance from the mean to an observation in the data. (How much of a spread in the data is there)

quartet.groupby('dataset').agg(["mean", "std"])
# Modeling

import statsmodels.api as sm



from scipy.stats import pearsonr



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
first = quartet[quartet.dataset == "I"]

X = first[["x"]]

y = first[["y"]]



first_linear_model = LinearRegression().fit(X, y)

first_intercept = first_linear_model.intercept_[0]

first_slope = first_linear_model.coef_[0][0]



# y = mx + b

print("Intercept is", first_intercept)

print("Slope is", first_slope)
# Fit the regression for the second dataset

second = quartet[quartet.dataset == "II"]

X = second[["x"]]

y = second[["y"]]



second_linear_model = LinearRegression().fit(X, y)

second_intercept = second_linear_model.intercept_[0]

second_slope = second_linear_model.coef_[0][0]



# Then make a new model for the third dataset

third = quartet[quartet.dataset == "III"]

X = third[["x"]]

y = third[["y"]]



third_linear_model = LinearRegression().fit(X, y)

third_intercept = third_linear_model.intercept_[0]

third_slope = third_linear_model.coef_[0][0]



# And finally fit the regression for the 4th dataset

forth = quartet[quartet.dataset == "IV"]

X = forth[["x"]]

y = forth[["y"]]



forth_linear_model = LinearRegression().fit(X, y)

forth_intercept = forth_linear_model.intercept_[0]

forth_slope = forth_linear_model.coef_[0][0]





first = {

    "dataset": "I",

    "slope": first_slope,

    "intercept": first_intercept

}

second = {

    "dataset": "II",

    "slope": second_slope,

    "intercept": second_intercept

}



third = {

    "dataset": "III",

    "slope": third_slope,

    "intercept": third_intercept

}



forth = {

    "dataset": "IV",

    "slope": forth_slope,

    "intercept": forth_intercept

}



output = pd.DataFrame([first, second, third, forth])

output
import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

x = np.linspace(0,40,40)

ax.spines['left'].set_position('center')

ax.spines['bottom'].set_position('center')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')

plt.plot(x, first_slope * x + first_intercept, '-r', label='first')

plt.plot(x, second_slope * x + second_intercept,'-.g', label='second')

plt.plot(x, third_slope * x + third_intercept,':b', label='third')

plt.plot(x, forth_slope * x + forth_intercept,'--m', label='forth')

plt.legend(loc='upper left')

plt.show()
print(output)
# Once you have compared the descriptive statistics of each of the datasets in the quartet,

# Uncomment the next line and run this cell to visualize all 4 datasets next to eachother!

# sns.lmplot(x="x", y="y", col="dataset", col_wrap=2, data=quartet,height=5, aspect=1.5);