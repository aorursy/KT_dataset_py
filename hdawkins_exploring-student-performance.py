import numpy as np

import pandas as pd

import seaborn as sns

import unicodecsv

import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr

from functools import partial

%matplotlib inline
schools = pd.read_csv('../input/school_data.csv')
schools.head()
attitudes = pd.read_csv('../input/response_data.csv')
attitudes.head()
sns.jointplot(x="Q1(%)", y="Q4(%)", data=attitudes);
sns.boxplot(data=attitudes);
sns.boxplot(data=attitudes.drop(['School ID'],1), orient = "h");
plt.figure(figsize=(8,7));

sns.violinplot(data=attitudes.drop(['School ID'],1), orient = "h");
all_info = schools.merge(attitudes, on = "School ID").drop(['School', 'Board ID', 'Num responses'],1)
all_info.head()
all_rearranged = pd.melt(all_info, id_vars=["School ID", "Board","Num students", "Level 1 (%)","Level 2 (%)", 

                                            "Level 3 (%)","Level 4 (%)", "Num F", "Num M" ], var_name="Question")
all_rearranged.head()
plt.figure(figsize=(9,7));

sns.swarmplot(x="Question", y ="value", data=all_rearranged, hue="Board", split = True);
all_rearranged["risk"] = np.where(all_rearranged['Level 1 (%)'] + all_rearranged['Level 2 (%)'] > 30, 'high', 'low')
all_rearranged.head()
plt.figure(figsize=(9,7));

sns.violinplot(x="Question", y ="value", data=all_rearranged, hue="risk", split = True);
compare = all_info.drop(['School ID','Board', 'Num students', 'Level 2 (%)', 'Level 3 (%)', 'Level 4 (%)', 'Num M', 'Num F'],1);

calc = partial(pearsonr,compare['Level 1 (%)'])
compare.apply(calc)
sns.jointplot(x="Level 1 (%)", y="Q3(%)", data=all_info);