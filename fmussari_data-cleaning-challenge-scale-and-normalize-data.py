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
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# manual mix-max scaling
# https://rasbt.github.io/mlxtend/user_guide/preprocessing/minmax_scaling/
# Scaled_i = (Xi-Xmin)/(Xmax-Xmin)
diff = original_data.max() - original_data.min()
scaled_data_manual = (original_data-original_data.min())/diff

# plot both together to compare
fig, ax=plt.subplots(1,3)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
sns.distplot(scaled_data_manual, ax=ax[2])
ax[2].set_title("Manual scaled data")
# To compare manual scaled with mix-max scaled data:
print(scaled_data[:10])
print(scaled_data_manual[:10])
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)
# stats.boxcox find the lambda that maximizes the log-likelihood function and return it as the second output argument.
# https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.boxcox.html
lmbda = normalized_data[1]
print(lmbda)

# stats.boxcox_normmax compute optimal Box-Cox transform parameter for input data.
lmbda = stats.boxcox_normmax(original_data,method='mle')  
# method='mle' minimizes the log-likelihood boxcox_llf. This is the method used in boxcox.
print(lmbda)

# Normalized_i = (Xi^lmbda - 1)/lmbda
normalized_data_manual = (original_data**lmbda-1)/lmbda

# plot both together to compare
fig, ax=plt.subplots(1,3)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
sns.distplot(normalized_data_manual, ax=ax[2])
ax[2].set_title("Manual norm. data")
# To compare manual normalized with stats.boxcox normalized data:
print(normalized_data[0][:10])
print(normalized_data_manual[:10])
def boxcox_normmax(x,brack=(-1.0,1.0)):
    N = len(x)
    # compute uniform median statistics
    Ui = zeros(N)*1.0
    Ui[-1] = 0.5**(1.0/N)
    Ui[0] = 1-Ui[-1]
    i = arange(2,N)
    Ui[1:-1] = (i-0.3175)/(N+0.365)
    # this function computes the x-axis values of the probability plot
    #  and computes a linear regression (including the correlation)
    #  and returns 1-r so that a minimization function maximizes the
    #  correlation
    xvals = distributions.norm.ppf(Ui)
    def tempfunc(lmbda, xvals, samps):
        y = boxcox(samps,lmbda)
        yvals = sort(y)
        r, prob  = stats.pearsonr(xvals, yvals)
        return 1-r
    return optimize.brent(tempfunc, brack=brack, args=(xvals, x))
# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
### YOOOOhhh

usd_goal_n = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
normalized_usd = stats.boxcox(usd_goal_n)[0]


# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_usd, ax=ax[1])
ax[1].set_title("Normalized data")



# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?

# select the goal column
g_goal = kickstarters_2017.goal

# scale the goals from 0 to 1
scaled_data = minmax_scaling(g_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?

i_positive_pledges = kickstarters_2017.pledged > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.pledged.loc[i_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")
