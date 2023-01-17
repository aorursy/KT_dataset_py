# 5 Day Data Challenge, Day 2

# Plot a Numeric Variable with a Histogram



import matplotlib.pyplot as plt

import pandas as pd



file = '../input/movies.csv'

data = pd.read_csv(file, encoding='latin1')



scores = data['score']



plt.title('Histogram of movie ratings')

plt.hist(scores)