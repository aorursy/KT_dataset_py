import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Reading the dataset
dataset = pd.read_csv('../input/weight-height/weight-height.csv')

# check the first 5 rows of the dataset
dataset.head()
# Checking the statistical features of the dataset
dataset.describe()
# Visualizing the dataset
dataset.plot(x='Height', y='Weight', style='o', ms=1)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.grid(True)
plt.show()
# x = our feature (Height)
# y = our target (Weight)
def estimate_coefficients(x, y):
    # calculating the number of data (observations) in our dataset
    n = np.size(x)
    
    # calculating the mean of x/feature (Height) and y/target (Weight)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # cross deviation and deviation about x/feature (Height)
    ss_xy = np.sum(y*x) - n*mean_y*mean_x
    ss_xx = np.sum(x*x) - n*mean_x*mean_x
    
    # calculation of the coefficients (theta_0 and theta_1)
    theta_1 = ss_xy/ss_xx
    theta_0 = mean_y - (theta_1*mean_x)
    
    # returning theta as a tuple of coffecients
    return(theta_0, theta_1)
# x = our feature (Height)
# y = our target (Weight)
# theta = our hypothesis coefficients (theta_0 and theta_1)
def regression_line(x, y, theta):
    # plotting the dataset
    plt.scatter(x, y, marker='o', s=1, color='b', alpha=0.6)
    plt.grid(True)
    
    # our Hypothesis (h)
    h = theta[0] + (theta[1] * x)
    
    # plotting the Regression Line
    plt.plot(x, h, color='r', alpha=0.6)
    
    # setting the figure labels
    plt.xlabel('x (Height)')
    plt.ylabel('y (Weight)')
    
    # plotting the figure
    plt.show()
# passing the Weight column and the Height column to the functions as x and y respectively
# passing passing the returning value of the estimate_coefficients function as theta to regression_line function
regression_line(x=dataset['Weight'], y=dataset['Height'], theta=estimate_coefficients(x=dataset['Weight'], y=dataset['Height']))
