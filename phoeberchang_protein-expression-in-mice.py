# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
mouse_data = pd.read_csv('../input/Data_Cortex_Nuclear.csv')
print(mouse_data.columns)

proteins = mouse_data.columns[1:78]
print(proteins)
for p in proteins:
    sns.boxplot(data = mouse_data, x = 'class', y = p)
    plt.show()
    
    ccs = mouse_data.loc[mouse_data['class'] == 'c-CS-s'][p]
    ccm = mouse_data.loc[mouse_data['class'] == 'c-CS-m'][p]
    tcs = mouse_data.loc[mouse_data['class'] == 't-CS-s'][p]
    tcm = mouse_data.loc[mouse_data['class'] == 't-CS-m'][p]
    css = mouse_data.loc[mouse_data['class'] == 'c-SC-s'][p]
    csm = mouse_data.loc[mouse_data['class'] == 'c-SC-m'][p]
    tss = mouse_data.loc[mouse_data['class'] == 't-SC-s'][p]
    tsm = mouse_data.loc[mouse_data['class'] == 't-SC-m'][p]
    print(stats.f_oneway(ccs, ccm, tcs, tcm))