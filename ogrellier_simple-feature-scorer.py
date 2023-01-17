# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
def gini(actual, pred):

    assert (len(actual) == len(pred))

    # Put actual, pred and index in a matrix

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    # Sort the matrix by multiple keys

    # first key is negative pred (sort descending)

    # second key is index

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    # Sum all actual values

    totalLosses = all[:, 0].sum()



    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.



    return giniSum / len(actual)





def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)
trn_df = pd.read_csv("../input/train.csv")

f_bins = [f for f in trn_df.columns if "_bin" in f]

f_cats = [f for f in trn_df.columns if "_cat" in f]

f_flts = [f for f in trn_df.columns if f not in f_bins + f_cats + ["target", "id"]]

f_all = [f_bins + f_cats + f_flts]
f_scores = pd.DataFrame(np.zeros((len(f_all), 2)), columns=["feature", "score"])

for i_f, f in enumerate(f_bins + f_cats + f_flts):

    # Get the score for each feature

    # If trn_df[f] has a negative score then - trn_df[f] has a positive one

    # Remember it's all about sorting !

    f_scores.loc[i_f] = [f, abs(gini_normalized(trn_df.target.values, trn_df[f].values))]

f_scores.sort_values(by="score", ascending=False, inplace=True)

f_scores.head(5)
# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(10, 20))



# Plot the total crashes

sns.set_color_codes("pastel")



sns.barplot(x="score", y="feature", 

            data=f_scores,

            palette=mpl.cm.ScalarMappable(cmap='viridis_r').to_rgba((100 * f_scores["score"] / .2)))

plt.xlabel("Score")

plt.ylabel("Feature")

plt.title("Raw Feature Gini Score")