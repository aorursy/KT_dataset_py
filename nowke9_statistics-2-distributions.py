import math

import numpy as np

import pandas as pd

from matplotlib import pyplot

from scipy import stats



matches    = pd.read_csv('../input/matches.csv')

deliveries = pd.read_csv('../input/deliveries.csv')
win_by_wickets_data = matches[matches.win_by_wickets > 0].win_by_wickets

win_by_wickets_freq = win_by_wickets_data.value_counts(sort=False)

print(win_by_wickets_freq)

plt = win_by_wickets_freq.plot.bar()

plt.set_title("Frequency distribution graph - Win by wickets")

plt.set_xlabel("Win by wickets")

plt.set_ylabel("Frequency")
win_by_wickets_rel_freq = win_by_wickets_data.value_counts(sort = False, normalize = True)

print(win_by_wickets_rel_freq)

plt = win_by_wickets_rel_freq.plot.bar()

plt.set_title("Relative Frequency distribution graph - Win by wickets")

plt.set_xlabel("Win by wickets")

plt.set_ylabel("Relative frequency (%)")
win_by_wickets_cumulative_freq = win_by_wickets_data.value_counts(sort = False, normalize = True).cumsum()

print(win_by_wickets_cumulative_freq)

plt = win_by_wickets_cumulative_freq.plot.bar()

plt.set_title("Cumulative relative frequency distribution graph - Win by wickets")

plt.set_xlabel("Win by wickets")

plt.set_ylabel("Cumulative relative frequency (%)")
plt = win_by_wickets_cumulative_freq.plot.line()

plt.axhline(y = win_by_wickets_cumulative_freq[6], xmax = 5.5/10, linestyle='dashed')

plt.axvline(x = 6, ymax = win_by_wickets_cumulative_freq[6], linestyle='dashed')
# Get mean (mu) and std (sigma)

win_by_wickets_mean, win_by_wickets_std = win_by_wickets_data.mean(), win_by_wickets_data.std()



# Plot histogram (normalized) - LIGHT-BLUE

win_by_wickets_data.hist(color='lightblue', weights = np.zeros_like(win_by_wickets_data) + 1.0 / win_by_wickets_data.count())



# Plot line graph - RED

win_by_wickets_data.value_counts(sort=False, normalize=True).plot.line(color='red')



# Normal distribution for random points between 1 to 10 with mean, std.

random_data = np.arange(1, 10, 0.001)

pyplot.plot(random_data, stats.norm.pdf(random_data, win_by_wickets_mean, win_by_wickets_std), color='green')
mu, sigma = 128, 25 # From the above example

highest_scores = np.random.normal(mu, sigma, 1000) # Random 1000 values



count, bins, _ = pyplot.hist(highest_scores, 100, normed = True, color='lightblue') # plot 100 points

pyplot.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *

    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),

    linewidth = 2, color = 'r') # Plot the PDF
win_by_runs_data = matches[matches.win_by_runs > 0].win_by_runs

win_by_runs_mean, win_by_runs_std = win_by_runs_data.mean(), win_by_runs_data.std()

z_score_35 = (35 - win_by_runs_mean) / win_by_runs_std

print(f'Z-score of 35 is {z_score_35:.2f}')
z_score = stats.norm.cdf(0.19)

print(f'z-score of 0.19 = {z_score * 100:.2f} percentile')
def compute_binomial_probability(x, n, p):

    """Returns the probability of getting `x` success outcomes in `n` trials,

    probability of getting success being `p`



    Arguments:



    x - number of trials of the event

    n - number of trials

    p - probability of the event



    """

    outcomes = math.factorial(n) / (math.factorial(x) * math.factorial(n - x))

    probability_of_each_outcome = p ** x * (1 - p) ** (n - x)

    return outcomes * probability_of_each_outcome



def plot_binomial_distribution_graph(n, p):

    """Plots Binomial distribution graph of an event with `n` trials,

    probability of getting success of the event being `p` for values `0` to `n`



    Arguments:



    n - number of trials

    p - probability of the event



    """

    probabilities = list(map(lambda x: compute_binomial_probability(x, n, p), range(0, n+1)))

    pyplot.bar(list(range(0, n+1)), probabilities)



plot_binomial_distribution_graph(5, 0.5)
plot_binomial_distribution_graph(10, 0.5)
plot_binomial_distribution_graph(10, 0.7)
pyplot.bar(['0', '1'], [0.35, 0.65])