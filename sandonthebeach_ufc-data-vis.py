# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
PATH = "/kaggle/input/ufcdata/"
data = pd.read_csv(PATH+'preprocessed_data.csv')
raw_data = pd.read_csv(PATH+'data.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
data.shape
df1 = data[data['R_age']<35]
df2 = data[data['R_age']>35]
df3 = data[data['R_age']<30]
df4 = data[data['R_age']>30]
df3['R_avg_opp_DISTANCE_att'].mean()
df4['R_avg_opp_DISTANCE_att'].mean()
df1['R_avg_SIG_STR_att'].mean()
R_avg_young = df1['R_avg_SIG_STR_att']
R_avg_old = df2['R_avg_SIG_STR_att']

R_avg_distant_young = df3['R_avg_opp_DISTANCE_att']
R_avg_distant_old = df4['R_avg_opp_DISTANCE_att']
from scipy.stats import ttest_ind
ttest_age_distant = ttest_ind(R_avg_distant_old, R_avg_distant_young)
ttest_age_distant
ttest_age_sg_str = ttest_ind(R_avg_young, R_avg_old)

ttest_age_sg_str
df2['R_avg_SIG_STR_att'].mean()
df1['R_avg_SIG_STR_att'].mean()
df2['R_avg_SIG_STR_att'].std()
selected = ['R_age', 'B_age', 'R_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_att', 'R_avg_opp_DISTANCE_landed',
           'B_avg_opp_DISTANCE_landed', 'B_avg_SIG_STR_att', 'R_avg_SIG_STR_att','B_avg_SIG_STR_landed',
           'R_avg_SIG_STR_landed']
important = data[selected]

modes = important.mode().transpose()
medians = important.median()

variances = important.var()
means = important.mean()
std = important.std()
result = pd.concat([std, means, variances, medians,modes], axis=1)
result
variances
means
import matplotlib.pyplot as plt
Age = important['B_age']
plt.hist(Age, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Distribution')
plt.xlabel('Age');
R_Distant_opp_att = data['R_avg_opp_DISTANCE_att']
plt.hist(Age, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Distribution')
plt.xlabel('Distant Strike Attempted by Opp onto R');
B_Distant_opp_att = data['B_avg_opp_DISTANCE_att']
plt.hist(Age, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Distribution')
plt.xlabel('Distant Strike Attempted by Opp onto B');
import seaborn as sns
selected1= ['R_age', 'B_age']
selected2= ['R_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_att', 'R_avg_opp_DISTANCE_landed',
           'B_avg_opp_DISTANCE_landed']
selected3= ['B_avg_SIG_STR_att', 'R_avg_SIG_STR_att','B_avg_SIG_STR_landed',
           'R_avg_SIG_STR_landed']
data[selected1].plot(kind='density', subplots=True, layout=(2, 1), 
                  sharex=False, figsize=(10, 8));
data[selected2].plot(kind='density', subplots=True, layout=(2, 2), 
                  sharex=False, figsize=(15, 8));
data[selected3].plot(kind='density', subplots=True, layout=(2, 2), 
                  sharex=False, figsize=(15, 8));
import seaborn as sns
corr_matrix = data[selected].corr()
sns.heatmap(corr_matrix);