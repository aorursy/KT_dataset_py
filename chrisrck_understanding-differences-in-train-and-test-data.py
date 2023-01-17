# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Thanks to yuliagm: https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read training dataset via chunks
train_reader = pd.read_csv('../input/train.csv', chunksize=18790470)
test = pd.read_csv('../input/test.csv')

#test['click_time'] = pd.to_datetime(test['click_time'])
#Analysis focuses on 5 variables
variables = ['ip', 'app', 'device', 'os', 'channel']

#Creating the sets that will contain the unique values in training_only, testing_only and that are shared by both training and testset
#In training and for shared values I differentiate between unique values that are "attributed" and those that aren't.
train_only_sets00 = [set([]) for _ in variables]
train_only_sets01 = [set([]) for _ in variables]
train_only_sets11 = [set([]) for _ in variables]

test_only_sets = [set([]) for _ in variables]

shared_sets00 = [set([]) for _ in variables]
shared_sets01 = [set([]) for _ in variables]
shared_sets11 = [set([]) for _ in variables]
#Creating the sets for all variables chunk by chunk
loaded_only = [set([]) for _ in variables]
notloaded_only = [set([]) for _ in variables]
both_states = [set([]) for _ in variables]

for cnum, chunk in enumerate(train_reader):
    print("Reading chunk %i" % cnum)
    app_loaded = chunk["is_attributed"] == 1
    app_notloaded = chunk["is_attributed"] == 0
    for vnum, v in enumerate(variables):
        train_notloaded = set(chunk[v][app_notloaded])
        train_loaded = set(chunk[v][app_loaded])
        
        loaded_diff = loaded_only[vnum] | (train_loaded - train_notloaded)
        notloaded_diff = notloaded_only[vnum]| (train_notloaded - train_loaded)
        
        chunk_carry = loaded_diff & notloaded_diff
        
        both_states[vnum] = both_states[vnum]| (train_notloaded & train_loaded) | chunk_carry
        loaded_only[vnum] = loaded_diff - both_states[vnum]
        notloaded_only[vnum] = notloaded_diff - both_states[vnum]
print("Finalizing sets")

for vnum, v in enumerate(variables):
    test_set = set(test[v])
    
    #Unique values of variable that are shared between training and test set
    shared_sets00[vnum] = test_set & notloaded_only[vnum]
    shared_sets01[vnum] = test_set & both_states[vnum]
    shared_sets11[vnum] = test_set & loaded_only[vnum]

    #Unique values of variable that are only in the test set
    test_only_sets[vnum] = test_set - shared_sets00[vnum] - shared_sets01[vnum] - shared_sets11[vnum]
    
    #Unique values of variable that are only in the training set
    train_only_sets00[vnum] = notloaded_only[vnum] - test_set
    train_only_sets01[vnum] = both_states[vnum] - test_set
    train_only_sets11[vnum] = loaded_only[vnum] - test_set

print("Done creating sets")
#DataFrame for analysis will consist of 5 columns, each for the subsets 1a, 1b, 2a, 2b, 3
#The rows of the dataframe are the number of unique items in the subset - one row per variable (ip, app...)

ana_data = {"shared_onlyloaded": list(map(len, shared_sets11)),
            "shared_both": list(map(len, shared_sets01)),
            "shared_noneloaded": list(map(len, shared_sets00)),
            "train_only_onlyloaded": list(map(len, train_only_sets11)),
            "train_only_both": list(map(len, train_only_sets01)),
            "train_only_noneloaded": list(map(len, train_only_sets00)),
            "test_only": list(map(len, test_only_sets))
           }

ana_frame = pd.DataFrame(data=ana_data)
#DataFrame is normalized towards total number of unique items
total = ana_frame.sum(axis=1)

for col in list(ana_frame):
    ana_frame[col] = ana_frame[col].divide(total)
from matplotlib import pyplot
#General function for analysis of the categorical variables

def analyze_column(col_num, col_name, bins, ignore_overview=False):
    #Is there an easier way to drop all but a particular row?
    inds = np.arange(5)
    inds_mask = np.ones(5, dtype=bool)
    inds_mask[col_num] = False
    ip_frame = ana_frame.drop(ana_frame.index[inds[inds_mask]])
    
    if not ignore_overview:
        #Plotting the initial overview
        ax = ip_frame.plot.bar()
        ax.set_xticklabels(variables)
        ax.set_title(col_name + "-Overview on items that are shared or differ between training and test set")

        for pnum, p in enumerate(ax.patches):
            ax.annotate(str(round(p.get_height()*total[col_num])), (p.get_x() * 0.99, p.get_height() * (1.01)))

    #plotting distribution
    pyplot.figure()
    pyplot.hist([list(shared_sets01[col_num]),list(shared_sets00[col_num]),list(shared_sets11[col_num]),
                list(test_only_sets[col_num]),list(train_only_sets01[col_num]), list(train_only_sets00[col_num]),
                list(train_only_sets11[col_num])], bins, alpha=1.0, 
                label=['shared_both', 'shared_noneloaded', 'shared_onlyloaded', 'test_only',
                       'train_only_both', 'train_only_noneloaded', 'train_only_onlyloaded'],
                stacked=True
               )
    pyplot.legend()
    pyplot.title(col_name + "-Analysis: Distribution of shared and differing unique values in training and test set")
    pyplot.show()
analyze_column(0, "IP", 50)
analyze_column(0, "IP", np.arange(125000, 128000, 50), True)
analyze_column(0, "IP", np.arange(123128, 123257, 2), True)
analyze_column(1, "APP", 20)
analyze_column(2, "Device", 50)
analyze_column(3, "OS", 50)
analyze_column(4, "channel", 30)