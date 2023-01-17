import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
wellbeing = pd.read_csv('/kaggle/input/lifestyle-and-wellbeing-data/Wellbeing_and_lifestyle_data.csv')

print(wellbeing.shape)

print(list(wellbeing.columns))

wellbeing.head()
wellbeing = wellbeing.drop('Timestamp', axis=1)

wellbeing = wellbeing.drop([10005]) # This entry contained errors that needed to be corrected or erased

age_dict = {'Less than 20' : 1, '21 to 35' : 2, '36 to 50' : 3, '51 or more' : 4}

wellbeing['AGE'] = pd.Series([age_dict[x] for x in wellbeing.AGE], index=wellbeing.index)

gender_dict = {'Female' : 1, 'Male' : 0}

wellbeing['GENDER'] = pd.Series([gender_dict[x] for x in wellbeing.GENDER], index=wellbeing.index)

wellbeing['DAILY_STRESS'] = wellbeing['DAILY_STRESS'].astype(int)
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white")



plt.figure(figsize=(10,4))

sns.distplot(wellbeing['ACHIEVEMENT'], hist=True, color="g").set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['DAILY_STRESS'], hist=True, rug=False, color="g").set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['DAILY_STEPS'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['FRUITS_VEGGIES'], hist=True, rug=False).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['PLACES_VISITED'], hist=True, rug=False).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['CORE_CIRCLE'], hist=True, rug=False).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['SUPPORTING_OTHERS'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['SOCIAL_NETWORK'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['DONATION'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['BMI_RANGE'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['TODO_COMPLETED'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['FLOW'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['LIVE_VISION'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['SLEEP_HOURS'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['LOST_VACATION'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['DAILY_SHOUTING'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['SUFFICIENT_INCOME'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['PERSONAL_AWARDS'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['TIME_FOR_PASSION'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['DAILY_MEDITATION'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['AGE'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
plt.figure(figsize=(10,4))

sns.distplot(wellbeing['GENDER'], hist=True).set_yticks([])

sns.despine(bottom=False, left=True)
wellcorr = wellbeing.corr().sort_values(by=['DAILY_STRESS'])

wellcorr[['ACHIEVEMENT', 'DAILY_STRESS']]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score
# An alternative method to gridsearch I was using to figure out the right hyperparameters to use

# Ultimately not as useful as I hoped



X = wellbeing.drop(['DAILY_STRESS', 'ACHIEVEMENT'], axis=1)

y = wellbeing[['DAILY_STRESS', 'ACHIEVEMENT']]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)

est_range = range(5, 105, 5)

score_graph = {'MAE Score': [],'R2 Score': [] , 'n_estimators': []}

est_list = []

score_list = []

for num in est_range:

    gbr = RandomForestRegressor(n_estimators=num, random_state=0)

    gbr.fit(train_X, train_y)

    ls_preds = gbr.predict(val_X)

    acc_1 = r2_score(val_y, ls_preds)

    score_graph['R2 Score'] = score_graph['R2 Score'] + [acc_1]

    acc_2 = mean_absolute_error(val_y, ls_preds)

    print(num ,'estimators \tR2 score is: ', acc_1,'\tMean Absolute Error is:', acc_2)

    score_graph['MAE Score'] = score_graph['MAE Score'] + [acc_2]

    score_graph['n_estimators'] = score_graph['n_estimators'] + [num]

    



fig, ax =plt.subplots(1,2, figsize=(20, 5))

ax[0].set_title('R2 Score by number of estimators')

sns.lineplot(x=score_graph['n_estimators'], y=score_graph['R2 Score'], ax=ax[0])

ax[1].set_title('Mean Absolute Error by number of estimators')

sns.lineplot(x=score_graph['n_estimators'], y=score_graph['MAE Score'], ax=ax[1]);
import warnings

warnings.filterwarnings("ignore")#silences one of the warning for eli5 about a future incompatability
import eli5

from eli5.sklearn import PermutationImportance



my_model = RandomForestRegressor(n_estimators=100).fit(train_X, train_y)



perm = PermutationImportance(my_model, n_iter=10, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
X = wellbeing.drop(['DAILY_STRESS'], axis=1)

y = wellbeing['DAILY_STRESS']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)

my_model = RandomForestRegressor(n_estimators=100).fit(train_X, train_y)



perm = PermutationImportance(my_model, n_iter=10, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
X = wellbeing.drop(['DAILY_STRESS', 'ACHIEVEMENT'], axis=1)

y = wellbeing[['DAILY_STRESS', 'ACHIEVEMENT']]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=.2)

my_model = RandomForestRegressor(n_estimators=100).fit(train_X, train_y)



perm = PermutationImportance(my_model, n_iter=10, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
_ = X.columns

df_pred = {}

print('For each of the following enter an integer value representing your answer to the survey questions')

for col in _:

    print('\n',col, end='\t')

    df_pred[col] = [int(input())]



df = pd.DataFrame.from_dict(df_pred, orient='columns')

df
_ = my_model.predict(df)

print('Prediction of Daily Stress: ', float(_[:,0]), "\t\tPrediction for Achievement: ", float(_[:,1]))