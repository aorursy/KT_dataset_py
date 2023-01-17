!wget -q 'https://www.dropbox.com/s/x0oxnc8es61z8ff/Reebok%20Ragnar%20-%20White%20Cliffs.xls?dl=1'
!wget -q 'https://www.dropbox.com/s/gc7vwijob4w73a0/Reebok%20Ragnar%20White%20Cliffs%202018.xls?dl=1'
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
last_rag_df = pd.read_excel('Reebok Ragnar - White Cliffs.xls?dl=1', header=1).assign(Year=2017)
last_rag_df['Time'] = pd.to_timedelta(last_rag_df['Time'], errors='coerce').dt.total_seconds()/3600
lr_df = last_rag_df.dropna()[['Name', 'Category', 'Time']].drop_duplicates()
lr_df.head(3)
this_rag_df = pd.read_excel('Reebok Ragnar White Cliffs 2018.xls?dl=1').assign(Year=2018)
this_rag_df['Time'] = pd.to_timedelta(this_rag_df['Time'], errors='coerce').dt.total_seconds()/3600
tr_df = this_rag_df.dropna()[['Name', 'Category', 'Time']].drop_duplicates()
tr_df.head(3)
comb_df = pd.merge(tr_df, lr_df, how='left', on=['Name'], suffixes=('_2018', '_2017')).dropna().copy()
imp_col_name = '2017->2018 Improvement (Hours)'
comb_df[imp_col_name] = (comb_df['Time_2018']-comb_df['Time_2017'])
comb_df.\
    drop(['Category_2018', 'Category_2017'], 1).\
    style.\
    bar(imp_col_name, align='mid', color=['#d65f5f', '#5fba7d']).\
    format({imp_col_name: '{:.1f}', 
            'Time_2018': '{:.1f}',
           'Time_2017': '{:.1f}'})
fig, ax1 = plt.subplots(1,1, figsize=(10, 5))
bins = np.linspace(20, 35, 12)
ax1.hist(tr_df['Time'], bins, label='2018', alpha=0.5)
ax1.hist(lr_df['Time'], bins, label='2017', alpha=0.5)
ax1.set_xlabel('Race Time (Hours)')
ax1.set_ylabel('Number of Teams')
ax1.legend();
from scipy.stats import ttest_ind_from_stats
ttest_ind_from_stats(tr_df['Time'].mean(), tr_df['Time'].std(), tr_df['Time'].shape[0], 
                     lr_df['Time'].mean(), lr_df['Time'].std(), lr_df['Time'].shape[0], equal_var=True)
import shutil
for c in glob('*.xls*'):
    shutil.move(c, '{}.xls'.format(os.path.splitext(c)[0]))
