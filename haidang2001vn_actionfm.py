!pip install turicreate
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import turicreate as tc

from datetime import datetime

from sklearn.model_selection import train_test_split
input_dir = '/kaggle/input/nailieactiondata/'

output_dir = '/kaggle/working/'
like = tc.SFrame(input_dir + 'fm_input_all_like.csv')

like.show()
follow = tc.SFrame(input_dir + 'fm_input_all_follow.csv')

follow.show()
dates = ['created_at', 'updated_at']

like_data = like

follow_data = follow
for date in dates:

    like_data[date] = like_data[date].str_to_datetime()

    follow_data[date] = follow_data[date].str_to_datetime()
def agg(target):

    target_like = like_data.groupby(target, operations={'action': tc.aggregate.COUNT, 'post': tc.aggregate.COUNT_DISTINCT('post_id')})

    target_like.rename({'action': 'like'}, inplace=True)

    

    target_follow = follow_data.groupby(target, operations={'action': tc.aggregate.COUNT})

    target_data = target_follow.join(target_like, on=target, how='left')

    

    target_data['like_post'] = target_data['like'] / target_data['post']

    target_data['follow_post'] = target_data['action'] / target_data['post']

    target_data['like_follow'] = target_data['like'] / target_data['action']

    

    columns = target_data.column_names()

    for col in columns[1:]:

        target_data[col] = target_data[col].apply(lambda val: np.log10(val))

    

    return target_data.dropna()
client_data = agg('client_id')

nailist_data = agg('nailist_id')
client_data.show()
nailist_data.show()
follow_data['hour'] = follow_data['created_at'].apply(lambda val: val.hour)

follow_data['weekday'] = follow_data['created_at'].apply(lambda val: val.weekday())

follow_data['month'] = follow_data['created_at'].apply(lambda val: val.month)
interaction_data = follow_data[['client_id', 'nailist_id']]

interaction_data.show()
pivot = int(interaction_data.shape[0] * 0.9)

interaction_train = interaction_data[:pivot]

interaction_valid = interaction_data[pivot:]
model = tc.ranking_factorization_recommender.create(num_factors=256, num_sampled_negative_examples=16, max_iterations=20,

                                                    observation_data=interaction_train, user_id='client_id', item_id='nailist_id',

                                                    user_data=client_data, item_data=nailist_data,

                                                    solver='auto', regularization=1e-6, random_seed=42)
results = model.evaluate_precision_recall(interaction_valid)
eval_result = results['precision_recall_by_user']

eval_result
K = 10

caller = {'precisionK': tc.aggregate.MEAN('precision'),

          'recallK': tc.aggregate.MEAN('recall'),

          'f1K': tc.aggregate.MEAN('f1'),

         }
eval_result['f1'] = 1/(1/eval_result['precision'] + 1/eval_result['recall'])

print(sorted(eval_result['cutoff'].unique()))

eval_result
eval_result[(eval_result['cutoff'] == K) & (eval_result['f1'] > 0)].show()
eval_result[eval_result['cutoff'] == K].show()
metricsK = eval_result[eval_result['cutoff'] <= K].groupby('client_id', operations=caller)

metricsK.show()
recommendation = model.recommend()

recommendation.show()
recommendation.export_csv(output_dir + 'FM_result.csv')

recommendation.print_rows(20*K)