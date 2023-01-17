import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
loadings_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv", index_col='Id')

scores_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv", index_col='Id')

scores_df['is_training'] = True

scores_df['site_id'] = '1'

combined_df = loadings_df.join(scores_df, how='outer')

combined_df.is_training.fillna(False, inplace=True)
sites_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv")

combined_df.loc[list(sites_df.Id), 'site_id'] = '2'

combined_df.site_id.fillna('1 or 2', inplace=True)

combined_df.site_id = combined_df.site_id.astype('category')
combined_df.site_id.value_counts().plot(kind='bar')
combined_df.query("is_training == True").site_id.value_counts().plot(kind='bar')
combined_df.query("is_training == False").site_id.value_counts().plot(kind='bar')
import seaborn as sns

import matplotlib.pylab as plt



for feature in ['IC_01', 'IC_07', 'IC_05', 'IC_16', 'IC_26', 'IC_06', 'IC_10', 'IC_09',

       'IC_18', 'IC_04', 'IC_12', 'IC_24', 'IC_15', 'IC_13', 'IC_17', 'IC_02',

       'IC_08', 'IC_03', 'IC_21', 'IC_28', 'IC_11', 'IC_20', 'IC_30', 'IC_22',

       'IC_29', 'IC_14']:

    for site in ['1', '2']:

        sns.distplot(combined_df.query('site_id == "%s"'%site)[feature], label='site %s'%site)

    plt.legend()

    plt.show()