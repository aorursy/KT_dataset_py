# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#! head ../input/forms.csv
horses = pd.read_csv('../input/horses.csv')

horseage = horses.groupby('age')['age', 'prize_money'].apply(lambda x: np.mean(x)).astype(int)

horseage['age'].corr(horseage['prize_money'])
horseage.plot()
horse_lookup = horseage.to_dict()['prize_money']

horses['weighted_prize_money'] = horses['prize_money'] - horses['age'].map(horse_lookup)
horses['quality'] = pd.qcut(horses['weighted_prize_money'], 10, labels=[1,2,3,4,5,6,7,8,9,10])



horses1 = pd.merge(horses, horses[['id', 'quality', 'sire_id', 'dam_id']], left_on=['sire_id'], right_on=['id'], suffixes=('', '_sire'))

horses1 = pd.merge(horses1, horses[['id', 'quality', 'sire_id', 'dam_id']], left_on=['dam_id'], right_on=['id'], suffixes=('', '_dam'))
# Show a subset of the dataframe

horses.plot(x='quality', y='weighted_prize_money')

#horses.head()
horses1['parents_quality'] = horses1[['quality_sire', 'quality_dam']].mean(axis=1)

horses1['quality'].corr(horses1['parents_quality'])

horses1.plot.scatter(x='quality', y='parents_quality')

plt.hexbin(horses1['quality'], horses1['parents_quality'], gridsize=10)
horses2 = pd.merge(horses1, horses1[['id', 'quality']], left_on=['sire_id_sire'], right_on=['id'], suffixes=('', '_sire_sire'))

horses2 = pd.merge(horses2, horses1[['id', 'quality']], left_on=['sire_id_dam'], right_on=['id'], suffixes=('', '_sire_dam'))



horses2 = pd.merge(horses2, horses1[['id', 'quality']], left_on=['dam_id_sire'], right_on=['id'], suffixes=('', '_dam_sire'))

horses2 = pd.merge(horses2, horses1[['id', 'quality']], left_on=['dam_id_dam'], right_on=['id'], suffixes=('', '_dam_dam'))
horses2['gparents_quality'] = horses2[['quality_sire',

                                      'quality_dam',

                                      'quality_sire_sire',

                                     'quality_sire_dam',

                                     'quality_dam_sire',

                                     'quality_dam_dam']].mean(axis=1)

horses2['quality'].corr(horses2['gparents_quality'])
horses2['gparents_quality'] = horses2[['quality_sire_sire',

                                     'quality_sire_dam',

                                     'quality_dam_sire',

                                     'quality_dam_dam']].mean(axis=1)

horses2['quality'].corr(horses2['gparents_quality'])
horses_low_var = horses2[horses2[['quality_sire',

         'quality_dam',

         'quality_sire_sire',

         'quality_sire_dam',

         'quality_dam_sire',

         'quality_dam_dam']].var(axis=1) < 10] #horses2



horses_low_var['gparents_quality'] = horses_low_var[['quality_sire',

                                                     'quality_dam',

                                                     'quality_sire_sire',

                                                     'quality_sire_dam',

                                                     'quality_dam_sire',

                                                     'quality_dam_dam']].mean(axis=1)

horses_low_var['quality'].corr(horses_low_var['gparents_quality'])