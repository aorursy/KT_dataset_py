import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
df_bottle = pd.read_csv("../input/bottle.csv")
df_cast = pd.read_csv("../input/cast.csv")

# set seed for reproducibility
np.random.seed(0)
df_bottle.info()
df_bottle['Salnty']
print(df_bottle.shape[0])
missing_values_count = df_bottle.isnull().sum()
missing_values_count
# select the  depth in meters, a linear parameter  that increases as we go down
depth_m = df_bottle.Depthm

# scale the Depth from 0 to 1
scaled_data = minmax_scaling(depth_m, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(df_bottle.Depthm, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


# get the index of all positive depth (Box-Cox only takes postive values)
index_of_positive_depth  = df_bottle.Depthm > 0

# get only positive depth (using their indexes)
positive_depth = df_bottle.Depthm.loc[index_of_positive_depth]

# normalize the depth (w/ Box-Cox)
normalized_depth = stats.boxcox(positive_depth)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_depth, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_depth, ax=ax[1])
ax[1].set_title("Normalized data")
#subset_nfl_data.fillna(0)
df_bottle_na_filled = df_bottle.fillna(0)
# select T_degC a linear parameter 

TdegC = df_bottle_na_filled.T_degC

# scale the T_degC from 0 to 1
scaled_data = minmax_scaling(TdegC, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(df_bottle_na_filled.T_degC, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# get the index of all positive T_degC (Box-Cox only takes postive values)
index_of_positive_T_degC  = df_bottle_na_filled.T_degC > 0

# get only positive T_degC (using their indexes)
positive_T_degC = df_bottle_na_filled.T_degC.loc[index_of_positive_T_degC]

# normalize the T_degC (w/ Box-Cox)
normalized_T_degC = stats.boxcox(positive_T_degC)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_T_degC, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_T_degC, ax=ax[1])
ax[1].set_title("Normalized data")
# select Salnty a linear parameter  

Salnty_m = df_bottle_na_filled.Salnty

# scale the T_degC from 0 to 1
scaled_data = minmax_scaling(Salnty_m, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(df_bottle_na_filled.Salnty, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
# get the index of all positive Salnty (Box-Cox only takes postive values)
index_of_positive_Salnty  = df_bottle_na_filled.Salnty > 0

# get only positive Salnty (using their indexes)
positive_Salnty = df_bottle_na_filled.Salnty.loc[index_of_positive_Salnty]

# normalize the Salnty (w/ Box-Cox)
normalized_Salnty = stats.boxcox(positive_Salnty)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_Salnty, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_Salnty, ax=ax[1])
ax[1].set_title("Normalized data")