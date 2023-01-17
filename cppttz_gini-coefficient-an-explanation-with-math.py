import numpy as np

import matplotlib.pyplot as plt

import scipy.interpolate

import scipy.integrate



predictions = np.array([0.9, 0.3, 0.8, 0.75, 0.65, 0.6, 0.78, 0.7, 0.05, 0.4, 0.4, 0.05, 0.5, 0.1, 0.1])

actual = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
def gini(actual, pred):

    assert (len(actual) == len(pred))

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)





def gini_normalized(actual, pred):

    return gini(actual, pred) / gini(actual, actual)
gini_predictions = gini(actual, predictions)

gini_max = gini(actual, actual)

ngini= gini_normalized(actual, predictions)

print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))
# Sort the actual values by the predictions

data = np.asarray(np.c_[actual, predictions,np.arange(len(actual))])

sorted_actual= data[np.lexsort((data[:,2],-1 * data[:, 1]))][:,0]

print('Sorted Actual Values', sorted_actual)

# Sum up the actual values

cumulative_actual = np.cumsum(sorted_actual)

cumulative_index = np.arange(1, len(cumulative_actual)+1)



plt.plot(cumulative_index, cumulative_actual)

plt.xlabel('Cumulative Number of Predictions')

plt.ylabel('Cumulative Actual Values')

plt.show()
cumulative_actual_shares = cumulative_actual / sum(actual)

cumulative_index_shares = cumulative_index / len(predictions)



# Add (0, 0) to the plot

x_values = [0] + list(cumulative_index_shares)

y_values = [0] + list(cumulative_actual_shares)



# Display the 45° line stacked on top of the y values

diagonal = [x - y for (x, y) in zip(x_values, y_values)]



plt.stackplot(x_values, y_values, diagonal)

plt.xlabel('Cumulative Share of Predictions')

plt.ylabel('Cumulative Share of Actual Values')

plt.show()

fy = scipy.interpolate.interp1d(x_values, y_values)

AUC,_ = scipy.integrate.quad(fy, 0, 1, points=x_values)

orange_area=AUC-0.5

print('Orange Area: %.3f' % orange_area)
cumulative_actual_shares_perfect = np.cumsum(sorted(actual,reverse=True)) / sum(actual)

y_values_perfect = [0] + list(cumulative_actual_shares_perfect)



# Display the 45° line stacked on top of the y values

diagonal = [x - y for (x, y) in zip(x_values, y_values_perfect)]



plt.stackplot(x_values, y_values_perfect, diagonal)

plt.xlabel('Cumulative Share of Predictions')

plt.ylabel('Cumulative Share of Actual Values')

plt.show()



# Integrate the the curve function

fy = scipy.interpolate.interp1d(x_values, y_values_perfect)

AUC,_ = scipy.integrate.quad(fy, 0, 1, points=x_values)

orange_area=AUC-0.5

print('Orange Area: %.3f' % orange_area)