!pip install ppscore
# importing the libraries

import os

import gc

import random

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import ipywidgets as widgets

import ppscore as pps





# setting the notebook parameters

root_dir = "/kaggle/input/lish-moa"

plt.rcParams["figure.figsize"] = (16, 8)

sns.set_style("darkgrid")

pd.set_option("display.max_rows", 20, "display.max_columns", None)
##########################

#### Helper Functions ####

##########################



def info_df(df):

    """

    returns a dataframe with number of unique values and nulls

    

    args: dataframe

    returns: dataframe

    """

    info = pd.DataFrame({

        "nuniques": df.nunique(),

        "% nuniques": round((df.nunique() / len(df) * 100), 2),

        "nulls": df.isnull().sum(),

        "% nulls": round((df.isnull().sum() / len(df) * 100), 2)

    })

    

    return info.T
# importing the datasets



train_df = pd.read_csv(os.path.join(root_dir, "train_features.csv"))

train_targets = pd.read_csv(os.path.join(root_dir, "train_targets_scored.csv"))

train_targets_nonscored = pd.read_csv(os.path.join(root_dir, "train_targets_nonscored.csv"))
train_df.head()
train_targets.head()
info_df(train_df)
info_df(train_targets)
pd.DataFrame({

    "% non-zero class - scored": round(train_targets.drop(columns = ["sig_id"]).sum() / len(train_targets) * 100, 2)

}).T
plt.subplot(1, 3, 1)

ax = sns.countplot(x = "cp_type", data = train_df)

for p in ax.patches:

    ax.annotate("{}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 150))   # for annotation of counts

plt.xlabel("CP Type")



plt.subplot(1, 3, 2)

ax = sns.countplot(x = "cp_time", data = train_df)

for p in ax.patches:

    ax.annotate("{}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 50))

plt.xlabel("CP Time")



plt.subplot(1, 3, 3)

ax = sns.countplot(x = "cp_dose", data = train_df)

for p in ax.patches:

    ax.annotate("{}".format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 100))

plt.xlabel("CP Dose")



plt.suptitle("CP - Features", fontsize = 20)

plt.show()
# list of gene expression columns

ge_list = [i for i in train_df.columns if i.startswith("g-")]



# feeding the above list to the dropdown

dropdown_ge_cols = widgets.Dropdown(options = ge_list)

ge_plot = widgets.Output()



def dropdown_ge_eventhandler(change):

    ge_plot.clear_output()

    with ge_plot:

        display(sns.distplot(train_df[change.new], kde = True, color = "g", label = change.new))

        plt.legend()

        plt.title("Density plot of {}".format(change.new))

        plt.show()

        

dropdown_ge_cols.observe(dropdown_ge_eventhandler, names = "value")
# display(dropdown_ge_cols)
# display(ge_plot)
sns.distplot(train_df["g-2"], kde = True, color = "g", label = "g-2")

plt.legend()

plt.title("Density plot of g-2")

plt.show()
# list of gene expression columns

cv_list = [i for i in train_df.columns if i.startswith("c-")]



# feeding the above list to the dropdown

dropdown_cv_cols = widgets.Dropdown(options = cv_list)

cv_plot = widgets.Output()



def dropdown_cv_eventhandler(change):

    cv_plot.clear_output()

    with cv_plot:

        display(sns.distplot(train_df[change.new], kde = True, color = "y", label = change.new))

        plt.legend()

        plt.title("Density plot of {}".format(change.new))

        plt.show()

        

dropdown_cv_cols.observe(dropdown_cv_eventhandler, names = "value")
# display(dropdown_cv_cols)
# display(cv_plot)
sns.distplot(train_df["c-2"], kde = True, color = "y", label = "c-2")

plt.legend()

plt.title("Density plot of c-2")

plt.show()
plt.subplots_adjust(hspace = 0.5)



plt.subplot(2, 2, 1)

sns.distplot(train_df["c-0"], kde = True)

plt.title("Distribution of C-0")



plt.subplot(2, 2, 2)

stats.probplot(train_df["c-0"], dist = "norm", plot = plt)



plt.subplot(2, 2, 3)

sns.distplot(train_df["c-1"], kde = True)

plt.title("Distribution of C-1")



plt.subplot(2, 2, 4)

stats.probplot(train_df["c-1"], dist = "norm", plot = plt)



plt.suptitle("Q-Q Plots of Cell Viability predictors")

plt.show()
# types of target variables

target_types = ["_inhibitor", "_agonist", "_antagonist", "_activator", "_blocker"]

col_counts = {} # key => the target column; val => number of such columns 

count = 0       # running count of how many columns are currently considered

for i in target_types:

    col_counts[i[1:]] = len([j for j in train_targets.columns if j.endswith(i)])

    count += col_counts[i[1:]]

col_counts["others"] = train_targets.shape[1] - count - 1 # -1 for sig_id column



# plot

sns.set_palette("hls")

bar_colors = ["r", "g", "b", "y", "m", "c"]

plt.bar(*zip(*col_counts.items()), color = bar_colors)

plt.title("Frequency plot of Targets types")

plt.show()
sns.catplot(x = "cp_type", col = "cp_time", hue = "cp_dose", data = train_df, kind = "count")

plt.suptitle("Interaction among the CP Variables")

plt.tight_layout(rect = [0, 0.03, 1, 0.95])

plt.show()
sns.color_palette("tab10")



g1 = sns.jointplot(x = "g-0", y = "g-1", data = train_df, kind = "reg", marker = "+")

g1.plot_marginals(sns.rugplot, height = -.15, clip_on = False)



plt.show()
g1 = sns.jointplot(x = "g-0", y = "g-1", data = train_df, kind = "kde", color = "b")

g1.plot_marginals(sns.rugplot, color = "b", height = -.15, clip_on = False)



plt.show()
# the ge_list used in the following line is defined in section 2.1.2

ppdf = pps.matrix(train_df[random.sample(ge_list, 10)])

ppdf = ppdf[["x", "y", "ppscore"]].pivot(columns = "x", index = "y", values = "ppscore")

sns.heatmap(ppdf, vmin = 0, vmax = 1, cmap = "Blues", linewidths = 0.5, annot = True)

plt.show()



del ppdf
g1 = sns.jointplot(x = "c-0", y = "c-1", data = train_df, kind = "reg", marker = "+")

g1.plot_marginals(sns.rugplot, height = -.15, clip_on = False)



plt.show()
g1 = sns.jointplot(x = "c-0", y = "c-1", data = train_df, kind = "kde", color = "b")

g1.plot_marginals(sns.rugplot, color = "b", height = -.15, clip_on = False)



plt.show()
# the cv_list used in the following line is defined in section 2.1.2

ppdf = pps.matrix(train_df[random.sample(cv_list, 10)])

ppdf = ppdf[["x", "y", "ppscore"]].pivot(columns = "x", index = "y", values = "ppscore")

sns.heatmap(ppdf, vmin = 0, vmax = 1, cmap = "Blues", linewidths = 0.5, annot = True)

plt.show()



del ppdf