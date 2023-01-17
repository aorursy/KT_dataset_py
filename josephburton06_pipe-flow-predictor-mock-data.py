import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import proof_of_concept_helpers

from proof_of_concept_helpers import create_pipe_data

from proof_of_concept_helpers import poly_regression



# Exploring

import scipy.stats as stats



# Modeling

import statsmodels.api as sm



from scipy.stats import pearsonr



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
pipe_1 = create_pipe_data(104)
pipe_1.head()
pipe_1.tail()
plt.scatter(pipe_1.days, pipe_1.percent_flow)
poly_regression(pipe_1, .8)
pipe_2 = create_pipe_data(76)
plt.scatter(pipe_2.days, pipe_2.percent_flow)
poly_regression(pipe_2, .8)
pipe_3 = create_pipe_data(142)
plt.scatter(pipe_3.days, pipe_3.percent_flow)
poly_regression(pipe_3, .8)
pipe_4 = create_pipe_data(211)
plt.scatter(pipe_4.days, pipe_4.percent_flow)
poly_regression(pipe_4, .8)
pipe_5 = create_pipe_data(21)
plt.scatter(pipe_5.days, pipe_5.percent_flow)
poly_regression(pipe_5, .8)