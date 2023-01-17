# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import forest

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
god_frame = pd.read_csv('../input/dump.csv')



god_frame['endDate'] = pd.to_datetime(god_frame['endDate'])

god_frame['startDate'] = pd.to_datetime(god_frame['startDate'])

god_frame['duration'] = (god_frame['endDate'] - god_frame['startDate']) / np.timedelta64(1, 'D')

god_frame = god_frame[god_frame['endDate'].notna()]

god_frame = god_frame[god_frame['duration'] > 0]

god_frame['hasPicture'] = -pd.isnull(god_frame['hasPicture']) * 1

god_frame['companyHasLogo'] = -pd.isnull(god_frame['companyHasLogo']) * 1



god_frame['companyUrl'] = -pd.isnull(god_frame['companyUrl']) * 1

god_frame['country'] = -pd.isnull(god_frame['country']) * 1



god_frame['title'] = god_frame['posTitle'] == god_frame['mbrTitle']

del god_frame['posTitle']

del god_frame['mbrTitle']



god_frame['location'] = god_frame['posLocation'] == god_frame['mbrLocation']

del god_frame['posLocation']

del god_frame['mbrLocation']



god_frame['locationCode'] = god_frame['posLocationCode'] == god_frame['mbrLocationCode']

del god_frame['posLocationCode']

del god_frame['mbrLocationCode']



god_frame['isMale'] = god_frame['genderEstimate'] == 'male'

god_frame['isFemale'] = god_frame['genderEstimate'] == 'female'

del god_frame['genderEstimate']



god_frame = god_frame.fillna(god_frame.mean())  # fill na with average value



del god_frame['memberUrn']

del god_frame['companyUrn']

del god_frame['positionId']

del god_frame['companyName']
mask = np.random.rand(len(god_frame)) < 0.6

train = god_frame.iloc[mask, :]

test = god_frame.iloc[~mask, :]

train_target = train.duration

# del train['daysToGo']

del train['startDate']

del train['endDate']

del train['duration']



# train.reset_index()

test_target = test.duration

# del test['daysToGo']

del test['startDate']

del test['endDate']

del test['duration']



dt_model = forest.RandomForestRegressor(n_estimators=100, max_depth=20, max_features=None)

# dt_model = joblib.load(f"{config.DIR['src']}/stats/compiled/r_forest_v1.pkl")



dt_model.fit(train, train_target)

# features = pd.DataFrame(dt_model.feature_importances_, index=train.columns, columns=['weight']).sort_values('weight')



result = pd.DataFrame(test_target.values, columns=['real'])

result['predicted'] = dt_model.predict(test)

result = result.sort_values('predicted', ascending=True)

result = result.reset_index()



print(f"Min: {np.min(result['real'])}, Max: {np.max(result['real'])}")

print(f"Average error of model: {np.average(np.abs(result['predicted'] - result['real']))}")  # 75

rand_vals = np.random.rand(len(result)) * np.max(result['real'])

# rand_vals = np.random.uniform(low=0.0, high=np.max(test_target), size=len(test_target))



print(f"Average error of random guesses: {np.average(np.abs(rand_vals - result['real']))}")  # 125



hi_idx = round(len(result) * 0.1)

med_idx = round(len(result) * 0.35)



result['pull'] = 'lo'

result.loc[hi_idx:med_idx, 'pull'] = 'med'

result.loc[:hi_idx, 'pull'] = 'hi'



print(np.mean(result[result['pull'] == 'lo']['real']))

print(np.mean(result[result['pull'] == 'med']['real']))

print(np.mean(result[result['pull'] == 'hi']['real']))