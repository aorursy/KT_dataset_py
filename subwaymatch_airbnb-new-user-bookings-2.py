from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
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
train = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')
test = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sample_submission_NDF.csv.zip')
sessions = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip')
display(train.head(3))
display(test.head(3))
display(sample_submission.head(3))
display(sessions.head(3))
test.shape
test.isna().sum()
alldata = pd.concat([train, test])
alldata.head(3)
sessions.rename(columns={
    'user_id': 'id'
}, inplace=True)

sessions
secs_elapsed_by_id = sessions.groupby('id', as_index=False)[['secs_elapsed']].mean()
secs_elapsed_by_id
alldata = alldata.merge(secs_elapsed_by_id, on='id', how='left')
alldata
plt.figure(figsize=(20, 12))

sns.boxplot(alldata['country_destination'], alldata['secs_elapsed'], showfliers=False)
sessions_nunique_by_id = sessions \
    .groupby('id')['action', 'action_type', 'action_detail','device_type'] \
    .nunique() \
    .reset_index()
    
sessions_nunique_by_id
alldata = alldata.merge(sessions_nunique_by_id, on='id', how='left')
alldata
id_count = sessions.groupby('id')['device_type'].count().to_frame().reset_index().rename(columns={
    'device_type': 'id_count'
})
id_count
alldata = alldata.merge(id_count, on='id', how='left')
alldata
alldata['action_ratio'] = alldata['action'] / alldata['id_count']
alldata['action_type_ratio'] = alldata['action_type'] / alldata['id_count']
alldata['action_detail_ratio'] = alldata['action_detail'] / alldata['id_count']
alldata['device_type_ratio'] = alldata['device_type'] / alldata['id_count']

alldata
date_account_created = pd.to_datetime(alldata['date_account_created'])

alldata['acc_create_year'] = date_account_created.dt.year
alldata['acc_create_month'] = date_account_created.dt.month
alldata['acc_create_day'] = date_account_created.dt.day
alldata['acc_create_dayofweek'] = date_account_created.dt.dayofweek

alldata.head()
date_account_first_active = pd.to_datetime(alldata['timestamp_first_active']
                                           .astype(str).str[:8], format='%Y%m%d')

alldata['first_active_year'] = date_account_first_active.dt.year
alldata['first_active_month'] = date_account_first_active.dt.month
alldata['first_active_day'] = date_account_first_active.dt.day
alldata['first_active_hour'] = alldata['timestamp_first_active'].astype(str).str[8:10]
alldata['first_active_dayofweek'] = date_account_first_active.dt.dayofweek

alldata.head(5)
alldata2 = alldata.drop(columns=['id',
                                 'date_account_created',
                                 'timestamp_first_active',
                                 'date_first_booking',
                                 'country_destination'])
alldata2.head(3)
alldata2 = alldata2.fillna(-1)
le = LabelEncoder()

for col in alldata2.columns[alldata2.dtypes == object]:
    # 그냥 돌리면 결측치가 있기 때문에 fit_transform()시에 오류가 난다
    # NaN을 하나의 카테고리로 만들어주기 위해 list()로 wrap하거나 astype(str) 사용
    
    alldata2[col] = le.fit_transform(list(alldata2[col]))
    # 혹은 alldata2[col] = le.fit_transform(alldata2[col].astype(str))
train2 = alldata2[:len(train)]
test2 = alldata2[len(train):]
# x_train, x_valid, y_train, y_valid = train_test_split(train2,
#                                                       train['country_destination'],
#                                                       test_size=0.2,
#                                                       random_state=42,
                                                      
#                                                       # country_destination 비율에 맞춰서 데이터를 뽑겠다
#                                                       stratify=train['country_destination']
#                                                      )
cbm = CatBoostClassifier(task_type='GPU', max_depth=7, iterations=1100)

cbm.fit(
    train2,
    train['country_destination'],
)

# 점수는 Log Loss로 나온다
# cbm = CatBoostClassifier(task_type='GPU')

# cbm.fit(
#     x_train,
#     y_train,
#     eval_set=(x_valid, y_valid),
#     early_stopping_rounds=30
# )

# # 점수는 Log Loss로 나온다
result = cbm.predict_proba(test2)
result
ids = []
country_list = []
le = LabelEncoder()

# inverse_transform()을 사용하면 나중에 숫자를 카테고리 라벨로 변환하기 쉽다
le.fit(train['country_destination'])
for i in range(len(sample_submission)):
    index = sample_submission['id'][i]
    
    ids += [index] * 5
    
    # 오늘의 핵심
    # 확률값이 가장 높은 애들을 가져온다\
    # [::-1]은 역순으로 가져온다는 의미
    country_list += le.inverse_transform(np.argsort(result[i])[::-1])[:5].tolist()
sub = pd.DataFrame({ 'id': ids, 'country': country_list })
sub
sub.to_csv('airbnb_new_user_bookings.csv', index=None)