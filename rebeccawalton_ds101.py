import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import operator

import statsmodels.api as sm

from IPython.core.display import display

import scipy.stats as ss

from matplotlib import pyplot as plt

from scipy.stats import binom



# This notebook will cover a couple of the ways to implement regression in python

# There are many libraries (and other languages!) that do this
# (1)

# Generate 'random' data similar to that on this slide

# Define the true values of the parameters (normally we wouldn't know these)

a_0 = 2

a_1 = 0.3

# Now generate some data from this distribution

np.random.seed(0)

X_1 = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5 - these will be our input or 'features'

res = 0.5 * np.random.randn(100)       # Generate 100 residual terms - this is the noise that can't be explained by our features

Y = a_0 + a_1 * X_1 + res                # Actual values of Y - this generates the results we observe in our test data



# The equation above (y = 2 + 0.3*x + noise) is the equation of the line shown in this slide.

# 2 is the intercept (where the red line crosses the y axis)

# 0.3 is the gradient



# The noise is the reason that the data points don't all sit exactly on the line

# Create pandas dataframe to store our X and y values

df = pd.DataFrame(

    {'X': X_1,

     'Y': y}

)

# Show the first five rows of our dataframe - if you were to plot these is would look similar to the graph on this slide

df.head()
# (2) - We generated our data above from the 'true' distribution and you can see how our code for this above

# (y = a_0 + a_1 * X1 + res) mirrors the equation on this slide

# however in real life we won't know the values of a_0 and a_1

# (2 and 0.3 respectively above) so we will need to estimate them from just the data in df
# (3) - Manual Maximum Likelihood Estimation for a binomial distribution (flipping a coin)

n = 10



# What are our chances of observing 10 heads if the coin is fair?

coin_prob_fair = 0.5

print('The probability of seeing these results from a fair coin is: ' + str(binom.pmf(10,10,coin_prob_fair)))



# What are our chances of observing 10 heads if the coin is biased?

coin_prob_biased = 1

print('The probability of seeing these results from a biased coin is: ' + str(binom.pmf(10,10,coin_prob_biased)))



hypoths_and_their_probabilities = {

    'fair' : binom.pmf(10,10,coin_prob_fair),

    'biased': binom.pmf(10,10,coin_prob_biased)

}



print('Therefore, it is most likely that the true probability of getting a head with this coin is: ' +

      max(hypoths_and_their_probabilities.items(), key=operator.itemgetter(1))[0])
# (4)

# Let's try out linear regression on our toy example from (1)

# Statsmodel doesn't automatically add an intercept or 'base value', so need to add this:

constant_column = np.repeat(1, 100)

X = pd.DataFrame(

    {'X_0': constant_column,

     'X_1': X_1}

)



model = sm.OLS(Y,X).fit()

print(model.summary())



# Under 'Method' you should see 'Least Squares' - this is the default cost function and is the one 

# discussed in the slides



# We haven't supplied the true parameters we used to generate this data to the model

# What does it think they are? You should be able to find the prediction for a_0 by looking in the 

# 'coef' column of the 'X_0' feature row and similarly for a_1.
# (5)

# How low did it manage to get the Mean Squared Error?

print(model.mse_resid)
# (6)

# Lets generate some more data, this time from a non-linear distribution

# Define the true values of the parameters (normally we wouldn't know these)

a_0 = 2

a_1 = 6

a_2 = 3

# Now generate some data from this distribution

np.random.seed(0)

X_1 = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5 - these will be our values for feature 1

X_2 = 1 * np.random.randn(100) + 5   # Array of 100 values with mean = 5, stddev = 12 - these will be our values for feature 2

res = 0.5 * np.random.randn(100)       # Generate 100 residual terms - this is the noise that can't be explained by our features

Y = a_0 + a_1 * X_1 + a_2 * np.square(X_2) + res                # Actual values of Y - this generates the results we observe in our test data



# How does a linear model do?

constant_column = np.repeat(1, 100)

X = pd.DataFrame(

    {'X_0': constant_column,

     'X_1': X_1,

     'X_2': X_2}

)



model = sm.OLS(Y,X).fit()

print(model.mse_resid)
# On my box there was an average squared error of about 12 with a linear model. Is it improved if we use

# a model with a quadratic term?

# First define our new feature X_3 as the square of X_2

X_3 = np.square(X_2)



# Now fit our model

X = pd.DataFrame(

    {'X_0': constant_column,

     'X_1': X_1,

     'X_2': X_2,

     'X_3': X_3}

)



model = sm.OLS(Y,X).fit()

print(model.mse_resid) 
# On my box this gives a much better result - an average squared error of about 0.2

# We can also see from looking at the summary that the coefficient for X_3 (our new quadratic feature)

# is much higher than that for X_2 which is now almost negligible as the quadratic term is identified as a

# much better predictor

print(model.summary())
# (7)

# If we decided to add many higher order terms as we did not know which had generated the underlying

# distribution, we might end up overfitting our model (too high variance)

# This might mean our results would generalise better if we fit a regularised model

X_4 = np.power(X_2, 3)

X_5 = np.power(X_2, 4)

X_6 = np.power(X_2, 5)

X_7 = np.power(X_2, 6)



# Now fit our model

constant_column = np.repeat(1, 100)

X = pd.DataFrame(

    {'X_0': constant_column,

     'X_1': X_1,

     'X_2': X_2,

     'X_3': X_3,

     'X_4': X_4,

     'X_5': X_5,

     'X_6': X_6,

     'X_7': X_7}

)



model = sm.OLS(Y,X).fit()

print(model.params)



model = sm.OLS(Y,X).fit_regularized()

print(model.params) 

# As expected,the higher order terms have decreased coefficients in the regularised model - which brings

# them much closer to the true value of 0.