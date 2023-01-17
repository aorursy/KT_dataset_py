import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



np.random.seed(123)
# gen data for proprtional sampling

raw_data = np.array([1,4,2,9,15,50,17])
# sort asc. order

data = np.array(sorted(raw_data))
sns.barplot(data, data)



plt.title("The probabiltiy of selecting each of the values is directlyproportional to it's value.\nWe can clearly see probabilty of selecting 50 is more \nthan probability of selecting -9 or 2 because it is greater\n\n data")

plt.grid()

plt.show()
# Normalize by divide by  sum (probability of selecting that value)

data_probs = data / data.sum()
# plot normalized / probabilites

sns.barplot(data, data_probs) # old-data: label, new-data: (data_scaled) obsvns



plt.grid()

plt.title("Normalized(div. by max): Probabilities\n Note, now w/ probabilites we can see what is probability of selecting each\n\n data_probs")

plt.show()
# cumulative sum

# --------------

# cumlative sums will be upperlimits of our

# uniform rvs to select that obsvc

labels = sorted(data)

data_norm_sorted = sorted(data_probs)

cum_sums = np.cumsum(data_norm_sorted)



cum_sums
pd.DataFrame({

    'Raw Input: data (sorted)': data,

    'Pobabilites: data_probs': data_probs,

    'Cumulative Sum: Upper Limits for uniform rvs': cum_sums

})
plt.figure(figsize=(10,7))

plt.plot(cum_sums)

plt.scatter(np.arange(0, len(cum_sums)), cum_sums)





y = cum_sums # sorted

x = data # sorted





plt. plot([-2, 0], [0, 0],linestyle="--")

for idx in range(0, len(cum_sums)):

    begx, begy = -2, cum_sums[idx]

    endx, endy = 6, cum_sums[idx]

    plt. plot([begx, endx], [begy, endy],linestyle="--")



_labels = ["Nothing"] + labels 

leg = [f"Lower-limit of {data[idx]} AND upper-limit of {_labels[idx]}" for idx in range(0,len(cum_sums+1))]



plt.legend(["Cumulative Sums"]+leg)



plt.title("Mapping U.R.V to Cumulative Sum For\nProportional Sampling\n\n" + \

         "The intervals of uniform r.v is given by dashed lines for proportional sampling of sorted `data`")

plt.ylabel("CDF")

plt.xlabel("indices of sorted sorted `data`")

plt.show()
# sampling function (w/ replacement)

def _sample_from_cum_sum(cum_sums, sample_size):

    proportional_sample_accumulator = []

    

    for _ in range(0, sample_size):

        urv = np.random.uniform(0,1)

        # sample based on uniform random variablE and upper-limits

        for idx, cum_sum in enumerate(cum_sums):

            if urv < cum_sum:

                porportionally_sampled_obsv = data[idx] # `data` is not raw (sorted above)

                proportional_sample_accumulator.append(porportionally_sampled_obsv)

                break # important

                

    return proportional_sample_accumulator
_sample_from_cum_sum(cum_sums, 3)