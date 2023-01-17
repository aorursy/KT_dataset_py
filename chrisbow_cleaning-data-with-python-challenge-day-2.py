# modules we'll use
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
kickstart = pd.read_csv("../input/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
indexOfPositivePledged = kickstart.pledged > 0

# get only positive pledges (using their indexes)
positivePledged = kickstart.pledged.loc[indexOfPositivePledged]

# normalise the pledges (w/ Box-Cox)
normalisedPledged = stats.boxcox(positivePledged)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positivePledged, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalisedPledged, ax=ax[1])
ax[1].set_title("Normalised Data")