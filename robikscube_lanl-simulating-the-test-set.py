import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

print(os.listdir("../input"))
samp_per_second = 31

ttf_array = np.concatenate([np.linspace(11, 0, samp_per_second * 11),

                np.linspace(11, 0, samp_per_second * 11),

                np.linspace(11, 0, samp_per_second * 11),

                np.linspace(5, 0, samp_per_second * 5),

                np.linspace(10, 0, samp_per_second * 10),

                np.linspace(15, 0, samp_per_second * 15),

                np.linspace(7, 0, samp_per_second * 7),

                np.linspace(15, 10, samp_per_second * 5)])

ss = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

ss_simulated = pd.DataFrame(ttf_array, columns=['time_to_failure'])

ss_simulated.plot(figsize=(15, 5))
ss_simulated.agg(['mean','median'])
sub1352 = pd.read_csv('../input/lanl-submissions/LB1.352_submission_147.csv')

sub1389 = pd.read_csv('../input/lanl-submissions/LB1.389_submissionverson_one.csv')

sub1362 = pd.read_csv('../input/lanl-submissions/LB1.362_submission_147.csv')

sub1371 = pd.read_csv('../input/lanl-submissions/LB1.371_submission_137.csv')

sub1380 = pd.read_csv('../input/lanl-submissions/LB1.380_submission.csv')

sub1381 = pd.read_csv('../input/lanl-submissions/LB1.381_submission_147.csv')

sub1389 = pd.read_csv('../input/lanl-submissions/LB1.389_submissionverson_one.csv')

sub_50feats = pd.read_csv('../input/lanl-submissions/submission_50feats_peaks 9000ks.csv')



ss['sub1.352'] = sub1352['time_to_failure']

ss['sub1.389'] = sub1389['time_to_failure']

ss['sub1.362'] = sub1362['time_to_failure']

ss['sub1.371'] = sub1371['time_to_failure']

ss['sub1.380'] = sub1380['time_to_failure']

ss['sub1.381'] = sub1381['time_to_failure']

ss['sub1.389'] = sub1389['time_to_failure']

ss['simulated'] = ss_simulated['time_to_failure']
ss_simulated.plot(kind='hist', bins= 100, figsize=(15, 5), alpha=0.5)

sub1352['time_to_failure'].plot(kind='hist', bins= 100, figsize=(15, 5), label='1.352 SUB', alpha=0.5)

sub1389['time_to_failure'].plot(kind='hist', bins= 100, figsize=(15, 5), label='1.389 SUB', alpha=0.5)

sub1362['time_to_failure'].plot(kind='hist', bins= 100, figsize=(15, 5), label='1.362 SUB', alpha=0.5)

plt.legend()

plt.show()
ss.agg(['mean','median'])
sub1389.agg(['mean','median'])
ss.sort_values('sub1.352').reset_index()['sub1.352'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')

ss.sort_values('sub1.389').reset_index()['sub1.389'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')

ss.sort_values('sub1.362').reset_index()['sub1.362'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')

ss.sort_values('sub1.371').reset_index()['sub1.371'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')

ss.sort_values('sub1.380').reset_index()['sub1.380'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')

ss.sort_values('sub1.381').reset_index()['sub1.381'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')

ss.sort_values('simulated').reset_index()['simulated'].plot(figsize=(15, 10), title='Compare TTF of simulated vs. Submissions')



plt.xlabel('Sample Sorted')

plt.ylabel('Time to Fault')

plt.legend()

plt.show()
sub1350_2037CV = pd.read_csv('../input/last-10-subs-data/1350_submission_lgb_FINALSHUFFLED_2.037CV_123Feats_24000Rows.csv')

sub1355_2038CV = pd.read_csv('../input/last-10-subs-data/1355_submission_lgb_FINALSHUFFLED_2.038CV_119Feats_24000Rows.csv')

sub1383_1991CV = pd.read_csv('../input/last-10-subs-data/1383_submission_lgb_FINALSHUFFLED_1.991CV_120Feats_24000Rows.csv')

sub1388_2086CV = pd.read_csv('../input/last-10-subs-data/1398_submission_lgb_FINALSHUFFLED_2.086CV_138Feats_24000Rows.csv')

sub1404_2009CV = pd.read_csv('../input/last-10-subs-data/1404_submission_lgb_FINALSHUFFLED_2.009CV_116Feats_24000Rows.csv')

sub1432_xgb = pd.read_csv('../input/last-10-subs-data/1432_submission_xgb_FINALSHUFFLED.csv')

sub1529_xgb = pd.read_csv('../input/last-10-subs-data/1529_submission_xgb_FINALSHUFFLED.csv')

sub1532_1948CV = pd.read_csv('../input/last-10-subs-data/1532_submission_lgb_1.948CV_110Feats_24000Rows_15Fold.csv')

sub1534_2092CV = pd.read_csv('../input/last-10-subs-data/1534_submission_lgb_FINALSHUFFLED_2.092CV_24Feats_24000Rows.csv')

sub1863_2283CV = pd.read_csv('../input/last-10-subs-data/1863_submission_lgb_FINALSHUFFLED_2.283CV_17Feats_24000Rows.csv')



ss['sub1350_2037CV'] = sub1350_2037CV['time_to_failure']

ss['sub1355_2038CV'] = sub1355_2038CV['time_to_failure']

ss['sub1383_1991CV'] = sub1383_1991CV['time_to_failure']

ss['sub1388_2086CV'] = sub1388_2086CV['time_to_failure']

ss['sub1404_2009CV'] = sub1404_2009CV['time_to_failure']

ss['sub1432_xgb'] = sub1432_xgb['time_to_failure']

ss['sub1529_xgb'] = sub1529_xgb['time_to_failure']

ss['sub1532_1948CV'] = sub1532_1948CV['time_to_failure']

ss['sub1534_2092CV'] = sub1534_2092CV['time_to_failure']

ss['sub1863_2283CV'] = sub1863_2283CV['time_to_failure']
ss.agg(['mean','median'])
submit = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
submit['time_to_failure'] = ss[['sub1.380','sub1.381','sub1529_xgb','sub1532_1948CV','sub1432_xgb']].mean(axis=1)
submit.agg(['mean','median'])
ss_simulated.plot(kind='hist', bins= 100, figsize=(15, 5), alpha=0.5)

submit['time_to_failure'].plot(kind='hist', bins= 100, figsize=(15, 5), alpha=0.5)

plt.show()
submit.to_csv('blend_5_including2xgb.csv', index=False)