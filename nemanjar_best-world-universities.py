import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
shanghai = pd.read_csv('../input/shanghaiData.csv')
print(shanghai.head())

# A bit of feature cleaning
shanghai.world_rank = [int(x.split('-')[0]) if type(x) == str else x for x in shanghai.world_rank]
TopN_last = shanghai[(shanghai.year == 2015) & (shanghai.world_rank.astype(int) < 81)]

TopNidxs = [True if name in TopN_last.university_name.unique() else False for name in shanghai.university_name]

TopN_all_yrs = shanghai[TopNidxs]
# Setting the plotting style 
sns.set_style('darkgrid')

# Auxiliary variable for neat annotation
label_occupancy = np.zeros(len(TopN_last))

for uni in TopN_all_yrs.university_name.unique():
    uni_df = TopN_all_yrs[TopN_all_yrs.university_name == uni]
    T = uni_df.year.unique()
    rank = uni_df.world_rank.values
    
    # Auxiliary variable for neat annotation
    lab_offs = -1
    
    while not(label_occupancy[rank[-1] + lab_offs] == 0):
        lab_offs += 1    
    
    label_occupancy[rank[-1] + lab_offs] = 1
    lab_y = rank[-1] + lab_offs + 1
    
    if lab_offs == -1:
         bullet = "("+ str(rank[-1]) + ") "
    else:
         bullet = " "*(5 + len(str(rank[-1])))           
    
    max_len = 30

    uni_name = (uni[:max_len] + '..') if len(uni) > max_len else uni
    
    plt.plot(T, rank, linewidth = 3)
    plt.text(T[-1]+0.2,
             lab_y,
             bullet + uni_name,
             verticalalignment='center')

    
plt.gca().invert_yaxis()
plt.xlim([2004, 2019])
fig = plt.gcf()
fig.set_size_inches(16.5, 30)
plt.show()