## pd and np to begin with
import pandas as pd
import numpy as np 

## old plots
import matplotlib.pyplot as plt
%matplotlib inline

## my sea horse seaborn
import seaborn as sns
sns.set()

### HTML hmm
from IPython.display import HTML

### Check the files
from os import listdir
print(listdir("../input"))

## supress those annyoing warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
train = pd.read_csv("../input/train.csv", nrows=10000000)

train.head(2)
train.rename({"acoustic_data" : "sig","time_to_failure" : "qtime"}, axis = "columns", inplace=True)
### lets see the values in such sensitive data
print(range(3))
for n in range(3):
    print(train.qtime.values[n])
fig, ax = plt.subplots(2,1, figsize=(20,12))

ax[0].plot(train.index.values, train.sig.values, c="blue")
ax[0].set_title("Sig of 10 M rows")
ax[0].set_xlabel("Index")
ax[0].set_ylabel("Signal");

ax[1].plot(train.index.values, train.qtime.values, c="green")
ax[1].set_title("Qtime of 10 M rows")
ax[1].set_xlabel("Index")
ax[1].set_ylabel("Qtime in ms");
fig, ax = plt.subplots(3,1,figsize=(20,18))
ax[0].plot(train.index.values[0:50000], train.qtime.values[0:50000], c="Red")
ax[0].set_xlabel("Index")
ax[0].set_ylabel("Time to quake")
ax[0].set_title("How does the second quaketime pattern look like?")
ax[1].plot(train.index.values[0:49999], np.diff(train.qtime.values[0:50000]))
ax[1].set_xlabel("Index")
ax[1].set_ylabel("Difference between quaketimes")
ax[1].set_title("Are the jumps always the same?")
ax[2].plot(train.index.values[0:4000], train.qtime.values[0:4000])
ax[2].set_xlabel("Index from 0 to 4000")
ax[2].set_ylabel("Quaketime")
ax[2].set_title("How does the quaketime changes within the first block?");
test_path = "../input/test/"
test_files = listdir("../input/test")
sample_submission = pd.read_csv("../input/sample_submission.csv")

fig, ax = plt.subplots(4,1, figsize=(20,25))

for n in range(4):
    seg = pd.read_csv(test_path  + test_files[n])
    ax[n].plot(seg.acoustic_data.values, c="Red")
    ax[n].set_xlabel("Index")
    ax[n].set_ylabel("Signal")
    ax[n].set_ylim([-300, 300])
    ax[n].set_title("Test {}".format(test_files[n]));
fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.distplot(train.sig.values, ax=ax[0], color="Green", bins=100, kde=False)
ax[0].set_xlabel("Signal")
ax[0].set_ylabel("Density")
ax[0].set_title("Signal distribution")

low = train.sig.mean() - 3 * train.sig.std()
high = train.sig.mean() + 3 * train.sig.std() 
sns.distplot(train.loc[(train.sig >= low) & (train.sig <= high), "sig"].values,
             ax=ax[1],
             color="Red",
             bins=150, kde=False)
ax[1].set_xlabel("Signal")
ax[1].set_ylabel("Density")
ax[1].set_title("Signal distribution without peaks");
stepsize = np.diff(train.qtime)
train = train.drop(train.index[len(train)-1])
train["stepsize"] = stepsize
train.head(5)
train.stepsize = train.stepsize.apply(lambda l: np.round(l, 10))
stepsize_counts = train.stepsize.value_counts()
stepsize_counts
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
### Rolling Window Approach 
window_sizes = [10, 50, 100, 1000]
for window in window_sizes:
    train["rolling_mean_" + str(window)] = train.sig.rolling(window=window).mean()
    train["rolling_std_" + str(window)] = train.sig.rolling(window=window).std()
fig, ax = plt.subplots(len(window_sizes),1,figsize=(20,6*len(window_sizes)))

n = 0
for col in train.columns.values:
    if "rolling_" in col:
        if "mean" in col:
            mean_df = train.iloc[4435000:4445000][col]
            ax[n].plot(mean_df, label=col, color="Green")
        if "std" in col:
            std = train.iloc[4435000:4445000][col].values
            ax[n].fill_between(mean_df.index.values,
                               mean_df.values-std, mean_df.values+std,
                               facecolor='Orange',
                               alpha = 0.5, label=col)
            ax[n].legend()
            n+=1
