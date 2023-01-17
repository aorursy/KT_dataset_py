import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Read the data set
data = [(12.079,19.278),
(16.791,18.741),
(9.564,21.214),
(8.630,15.687),
(14.669,22.803),
(12.238,20.878),
(14.692,24.572),
(8.987,17.394),
(9.401,20.762),
(14.480,26.282),
(22.328,24.524),
(15.298,18.644),
(15.073,17.510),
(16.929,20.330),
(18.200,35.255),
(12.130,22.158),
(18.495,25.139),
(10.639,20.429),
(11.344,17.425),
(12.369,34.288),
(12.944,23.894),
(14.233,17.960),
(19.710,22.058),
(16.004,21.157),]
labels = ['Congruent', 'Incongruent']

df = pd.DataFrame.from_records(data, columns=labels)

# Print the shape of the data
print("Shape of the DataFrame is -> " + str(df.shape))

# Show the data
df.head(n=24)
# Describe the data
df.describe()
# Print measures of central tendency
# mean
print("Measures of central tendency")
print("-----------")
mean_congruent = np.mean(df['Congruent'])
mean_incongruent = np.mean(df['Incongruent'])

print("Mean of Congruent Data -> " + str(mean_congruent))
print("Mean of Incongruent Data -> " + str(mean_incongruent))
print("\n")

# median
median_congruent = np.median(df['Congruent'])
median_incongruent = np.median(df['Incongruent'])

print("Median of Congruent Data -> " + str(median_congruent))
print("Median of Incongruent Data -> " + str(median_incongruent))
print("\n")

# Print measures of variability
# range
print("Measures of Variability")
print("-----------")
range_congruent = np.ptp(df['Congruent'])
range_incongruent = np.ptp(df['Incongruent'])

print("Range of Congruent Data -> " + str(range_congruent))
print("Range of Incongruent Data -> " + str(range_incongruent))
print("\n")

# standard deviation
std_congruent = np.std(df['Congruent'], ddof = 1)
std_incongruent = np.std(df['Incongruent'], ddof = 1)

print("Standard Deviation of Congruent Data -> " + str(std_congruent))
print("Standard Deviation of Incongruent Data -> " + str(std_incongruent))
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df['Congruent'], bins = 35, color='salmon', alpha=0.5)
ax.hist(df['Incongruent'], bins = 35, color='lightblue', alpha=0.5)

ax.set(title='Comparing Congruent and Incongruent Response Times', ylabel='No. of Observations', xlabel = 'Response Time in Sec')
ax.margins(0.05)
ax.set_ylim(bottom=0)
plt.show()
fig, ax = plt.subplots(figsize=(10, 6))

bp = plt.boxplot([df['Congruent'], df['Incongruent']], vert = False, widths = 0.25, labels = ['Congruent', 'Incongruent'])

ax.set(title='Comparing Congruent and Incongruent Response Times', ylabel='Features', xlabel = 'Response Time in Sec')
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
plt.setp(bp['boxes'], color='blue')
plt.setp(bp['whiskers'], color='blue')
plt.setp(bp['fliers'], color='green', marker='+')
plt.show()