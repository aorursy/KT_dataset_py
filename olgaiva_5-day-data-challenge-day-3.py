import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plotting
import scipy.stats as stats # Stats, used for t-test

# Read in museum file
museum_df = pd.read_csv("../input/museums.csv")

colonies = ["ME","NH","VT","NY","CT","RI","MA","PA","NJ","DE","MD","VA","WV","NC","SC","GA","DC"]

# Throw out rows which don't have relevant parameters listed
museum_df = museum_df.dropna(axis=0,subset=["State (Administrative Location)","Museum Type"])
states = museum_df["State (Administrative Location)"]
states_by_val = dict.fromkeys(states)
for state in states_by_val.keys():
    state_museum_df = museum_df[museum_df["State (Administrative Location)"]==state]
    state_history_museum_df = state_museum_df[state_museum_df["Museum Type"].isin(["HISTORY MUSEUM", "HISTORIC PRESERVATION"])]
    states_by_val[state] = len(state_history_museum_df)/len(state_museum_df)

colonies_vals = [float(value) for (key, value) in states_by_val.items() if key in colonies]
state_vals = [float(value) for value in states_by_val.values()]
pop_mean = np.sum(state_vals)/len(state_vals)
print(pop_mean)
colonies_mean = np.sum(colonies_vals)/len(colonies_vals)
print(colonies_mean)
states_by_val_HtoL = sorted(states_by_val.items(), key=lambda x: x[1], reverse=True)
print(states_by_val_HtoL)
plt.hist(state_vals, bins=7)
plt.xlabel("Ratio of history museums")
plt.ylabel("Number of states")
plt.show()
# For a population of this size I think this is OK

plt.boxplot(state_vals)
plt.show()
test_result1 = stats.ttest_1samp(colonies_vals, pop_mean)
print(test_result1)

# Out of curiosity, what happens if I throw out the outliers?
states_by_val.pop("UT")
states_by_val.pop("DC")
colonies_vals = [float(value) for (key, value) in states_by_val.items() if key in colonies]
state_vals = [float(value) for value in states_by_val.values()]
test_result2 = stats.ttest_1samp(colonies_vals, pop_mean)
print(test_result2)