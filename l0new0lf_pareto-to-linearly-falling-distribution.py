import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# gen data

pareto_data = np.random.pareto(a=2, size=1000)

ys = pareto_data

xs = np.array([f"w{i}" for i in range(0, len(ys))])



# sort (pareto will be visible only like 

# by doing so becuase x is categorical data)

ys_sorted_idxs = np.argsort(ys)[::-1]

ys_sorted = ys[ys_sorted_idxs]

xs_sorted = xs[ys_sorted_idxs]



# plot

plt.figure(figsize=(20, 8))

sns.barplot(xs_sorted, ys_sorted)

#plt.xticks(rotation=70)

plt.xlabel("Separate individual words (Categorical Data)")

plt.ylabel("Word counts")

plt.show()
plt.figure(figsize=(20, 6))

sns.barplot(xs_sorted[:100], ys_sorted[:100])

plt.xticks(rotation=70)

plt.xlabel("Separate individual words (Categorical Data)")

plt.ylabel("Word counts")

plt.show()
# transform one of axis to logs

ys_sorted_logs = np.log(ys_sorted)
# plot

plt.figure(figsize=(20, 8))

sns.barplot(xs_sorted, ys_sorted_logs)

#plt.xticks(rotation=70)

plt.xlabel("Separate individual words (Categorical Data)")

plt.ylabel("Log(Word counts)")

plt.show()
plt.figure(figsize=(20, 14))

sns.barplot(xs_sorted[:100], ys_sorted_logs[:100])

plt.xticks(rotation=70)

plt.xlabel("Separate individual words (Categorical Data)")

plt.ylabel("Log(Word counts)")

plt.show()