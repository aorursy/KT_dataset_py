import pandas as pd

train = pd.read_csv("/kaggle/input/hse-pml-2/train_resort.csv")

test = pd.read_csv("/kaggle/input/hse-pml-2/test_resort.csv")

train['amount_spent_per_room_night_scaled'].hist()
print(test['season_holidayed_code'].unique())

print(train['state_code_residence'].unique())

print(test['season_holidayed_code'].unique())

print(train['state_code_residence'].unique())
import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 12





plt.subplot(1, 2, 2)

sns.distplot(train['season_holidayed_code'], color='red')

plt.title('season_holidayed_code train distribution')



plt.subplot(1, 2, 2)

sns.distplot(test['season_holidayed_code'], color='green')

plt.title('season_holidayed_code test distribution')



plt.subplot(1, 2, 2)

sns.distplot(train['state_code_residence'], color='red')

plt.title('season_holidayed_code train distribution')



plt.subplot(1, 2, 2)

sns.distplot(test['state_code_residence'], color='green')

plt.title('season_holidayed_code test distribution')
plt.rcParams['figure.figsize'] = (18, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 2)

sns.scatterplot(x=train['state_code_residence'], y=train['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')



#data['state_code_residence'].value_counts()
plt.rcParams['figure.figsize'] = (15, 6)

plt.rcParams['font.size'] = 12



plt.subplot(1, 2, 1)

sns.countplot(train['member_age_buckets'], color = 'red')

sns.countplot(test['member_age_buckets'], color = 'green')





plt.title('Distribution of member_age_buckets')



plt.subplot(1, 2, 2)

sns.lineplot(x=train['member_age_buckets'], y=train['amount_spent_per_room_night_scaled'])

plt.title('Amount spent for a room per night')