# First of all I have to thank ArjoonnSharma
# I have learned a lot from the script Preliminary Exploration
# this script is largely based on his/her work, and I hope it's OK

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../input/data.csv")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
def test_it(data_test):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=5)  # A simple classifier
    return cross_val_score(clf, data_test.drop('shot_made_flag', 1), data_test.shot_made_flag,
                           scoring='accuracy', cv=10
                          )
# define the sort & enumeration function
def sort_encode(df, field):
    ct = pd.crosstab(df.shot_made_flag, df[field]).apply(lambda x:x/x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    df[new_field] = df[field].map(new_map)
    return new_field
auc_list = {}
# action_type
new_field = sort_encode(data, 'action_type')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# combined_shot_type
new_field = sort_encode(data, 'combined_shot_type')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# season
data['season_start_year'] = data.season.str.split('-').str[0]
data['season_start_year'] = data['season_start_year'].astype(int)

new_field = 'season_start_year'
data_test = data[['season_start_year', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_distance
new_field = 'shot_distance'
data_test = data[['shot_distance', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_type (2 or 3 points)
new_field = sort_encode(data, 'shot_type')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_zone_area
new_field = sort_encode(data, 'shot_zone_area')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_zone_basic
new_field = sort_encode(data, 'shot_zone_basic')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
# shot_zone_range
new_field = sort_encode(data, 'shot_zone_range')
data_test = data[[new_field, 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
data['xy'] = np.sqrt(data['loc_x']*data['loc_x'] + data['loc_y']*data['loc_y'])
data_test = data[['xy', 'shot_made_flag']].copy()
data_test = data_test.dropna()
auc_mean = test_it(data_test).mean()
auc_list[new_field] = auc_mean
print(auc_mean)
auc_list
f1, f2 = 'action_type_sort_enumerated', 'shot_zone_range_sort_enumerated'
f3 = 'xy'
data_test = data[[f1, f2, f3, 'shot_made_flag']].copy()
data_test = data_test.dropna()
test_it(data_test).mean()
clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=7, random_state=2016) # a more powerful classifier

f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_zone_range_enumerated_sort_enumerated'
f3 = 'home_or_away'
train = data.loc[~data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_made_flag']]
test = data.loc[data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_id']]

# Impute
mode = test.action_type_enumerated_sort_enumerated.mode()[0]
test.action_type_enumerated_sort_enumerated.fillna(mode, inplace=True)

# Train and predict
clf.fit(train.drop('shot_made_flag', 1), train.shot_made_flag)
predictions = clf.predict_proba(test.drop('shot_id', 1))

# convert to CSV
submission = pd.DataFrame({'shot_id': test.shot_id,
                           'shot_made_flag': predictions[:, 1]})
submission[['shot_id', 'shot_made_flag']].to_csv('submission.csv', index=False)
clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=7, random_state=2016) # a more powerful classifier

f1, f2 = 'action_type_enumerated_sort_enumerated', 'shot_distance'
f3 = 'home_or_away'
train = data.loc[~data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_made_flag']]
test = data.loc[data.shot_made_flag.isnull(), [f1, f2, f3, 'shot_id']]

# Impute
mode = test.action_type_enumerated_sort_enumerated.mode()[0]
test.action_type_enumerated_sort_enumerated.fillna(mode, inplace=True)

# Train and predict
clf.fit(train.drop('shot_made_flag', 1), train.shot_made_flag)
predictions = clf.predict_proba(test.drop('shot_id', 1))

# convert to CSV
submission = pd.DataFrame({'shot_id': test.shot_id,
                           'shot_made_flag': predictions[:, 1]})
submission[['shot_id', 'shot_made_flag']].to_csv('submission.csv', index=False)