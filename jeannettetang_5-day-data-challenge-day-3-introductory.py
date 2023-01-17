import pandas as pd



data = pd.read_csv("../input/cereal.csv")

data.head()
# sodium for two groups (hot, cold)

cold = data["sodium"][data["type"] == "C"]

hot = data["sodium"][data["type"] == "H"]
from numpy import std



# checks if standard deviation of sodium for hot cereal is equal to

# standard deviation of sodium for cold cereal

eq_var = (std(cold) == std(hot))
from scipy.stats import ttest_ind



ttest_ind(hot, cold, equal_var = eq_var) # runs t-test and returns p-value
# histogram for cold group

cold.hist()
# histogram for hot group

hot.hist()