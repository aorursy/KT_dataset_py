import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
seeds_0_5 = pd.read_csv('../input/skip-rnn-cv-meta-shakeup-seeds/shakeup.csv')
seeds_5_10 = pd.read_csv('../input/skip-rnn-cv-meta-shakeup-seeds-5-10/shakeup.csv')
seeds_10_15 = pd.read_csv('../input/skip-rnn-cv-meta-shakeup-10-15/shakeup.csv')
seeds_15_20 = pd.read_csv('../input/skip-rnn-cv-meta-shakeup-15-20/shakeup.csv')
shakeup_df = pd.concat([seeds_0_5, seeds_5_10, seeds_10_15, seeds_15_20], ignore_index=True)
shakeup_df['val-stage1_shakeup'] = shakeup_df['stage_1_test_f1'] - shakeup_df['val_f1']
shakeup_df['val-stage2_shakeup'] = shakeup_df['stage_2_test_f1'] - shakeup_df['val_f1']
shakeup_df['stage1_stage2_shakeup'] = shakeup_df['stage_2_test_f1'] - shakeup_df['stage_1_test_f1']
shakeup_df.head()
print('Mean and Standard Deviation')
print('Stage 1 Val F1:\n ', round(shakeup_df['val_f1'].mean(), 6), round(shakeup_df['val_f1'].std(), 6))
print('Stage 1 Test F1:\n ', round(shakeup_df['stage_1_test_f1'].mean(), 6), round(shakeup_df['stage_1_test_f1'].std(), 6))
print('Stage 2 Test F1:\n ', round(shakeup_df['stage_2_test_f1'].mean(), 6), round(shakeup_df['stage_2_test_f1'].std(), 6))
print('Val - Stage 1 Shakeup:\n ', round(shakeup_df['val-stage1_shakeup'].mean(), 6), round(shakeup_df['val-stage1_shakeup'].std(), 6))
print('Val - Stage 2 Shakeup:\n ', round(shakeup_df['val-stage2_shakeup'].mean(), 6), round(shakeup_df['val-stage2_shakeup'].std(), 6))
print('Stage 1 - Stage 2 Shakeup:\n ', round(shakeup_df['stage1_stage2_shakeup'].mean(), 6), round(shakeup_df['stage1_stage2_shakeup'].std(), 6))
plt.hist(shakeup_df['threshold'])
plt.show()
sns.kdeplot(shakeup_df['threshold'])
plt.show()
plt.hist(shakeup_df['val_f1'])
plt.show()
sns.kdeplot(shakeup_df['val_f1'])
plt.show()
plt.hist(shakeup_df['stage_1_test_f1'])
plt.show()
sns.kdeplot(shakeup_df['stage_1_test_f1'])
plt.show()
plt.hist(shakeup_df['stage_2_test_f1'])
plt.show()
sns.kdeplot(shakeup_df['stage_2_test_f1'])
plt.show()
plt.hist(shakeup_df['val-stage1_shakeup'])
plt.show()
sns.kdeplot(shakeup_df['val-stage1_shakeup'])
plt.show()
plt.hist(shakeup_df['val-stage2_shakeup'])
plt.show()
sns.kdeplot(shakeup_df['val-stage2_shakeup'])
plt.show()
plt.hist(shakeup_df['stage1_stage2_shakeup'])
plt.show()
sns.kdeplot(shakeup_df['stage1_stage2_shakeup'])
plt.show()







