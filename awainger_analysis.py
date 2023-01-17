import feather
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import linregress

from scipy.stats import ttest_ind
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
# Loading all the datasets I built in the preprocessing notebook
play_information = feather.read_dataframe('../input/nflpuntpreprocessed/play_information.feather')
game_data = feather.read_dataframe('../input/nflpuntpreprocessed//game_data.feather')
play_player_role_data = feather.read_dataframe('../input/nflpuntpreprocessed/play_player_role_data.feather')
min_distances = feather.read_dataframe('../input/nflpuntpreprocessed/min_distances.feather')
second_min_distances = feather.read_dataframe('../input/nflpuntpreprocessed/second_min_distances.feather')
punt_hangtime = feather.read_dataframe('../input/nflpuntpreprocessed/punt_hangtime.feather')
video_review = feather.read_dataframe('../input/nflpuntpreprocessed/video_review.feather')
# Building a master play-level dataset
aug_play_information = play_information.merge(
    min_distances[['GameKey', 'PlayID', 'Coverage_Distance']],
    on=['GameKey', 'PlayID'], how='left', validate='one_to_one'
).merge(
    second_min_distances[['GameKey', 'PlayID', 'Coverage_Distance']],
    on=['GameKey', 'PlayID'], how='left', validate='one_to_one',
    suffixes=['_1', '_2']
).merge(
    game_data[['GameKey', 'Temperature', 'Is_Grass', 'Is_Outdoor']] ,
    on=['GameKey'], how='left', validate='many_to_one'
).merge(
    punt_hangtime[['GameKey', 'PlayID', 'Hangtime']],
    on=['GameKey', 'PlayID'], how='left', validate='one_to_one'
).merge(
    video_review,
    on=['GameKey', 'PlayID'], how='left', validate='one_to_one',
)

aug_play_information['Has_Concussion'] = ~aug_play_information.GSISID.isna()
plays_with_punts = aug_play_information[aug_play_information.Has_Punt]
plays_with_concussions = aug_play_information[aug_play_information.Has_Concussion]

print("Total plays:", aug_play_information.shape[0])
print("Plays with a punt:", plays_with_punts.shape[0])
print("Plays with a concussion:", plays_with_concussions.shape[0])
print("Plays with a punt and a concussion:", plays_with_concussions[plays_with_concussions.Has_Punt].shape[0])
plays_with_concussions = plays_with_concussions[plays_with_concussions.Has_Punt]

concussion_rates = plays_with_concussions.Punt_Type.value_counts() / plays_with_punts.Punt_Type.value_counts()
ax = concussion_rates.plot(
    kind='barh', figsize=(10,5), color='#2678B2', fontsize=12,
    title='Concussions Occur Over 7x More Often On Returned Punts')
vals = ax.get_xticks()
ax.set(xlabel="Concussion Rate", xticklabels=['{:,.1%}'.format(x) for x in vals]);
ax.xaxis.label.set_size(14)
ax.title.set_size(16)

ttest_res = ttest_ind(
    plays_with_punts[plays_with_punts.Has_Return].Has_Concussion, 
    plays_with_punts[~plays_with_punts.Has_Return].Has_Concussion
)

if ttest_res.pvalue < .05:
    print('The difference in concussion rates on plays with and without returns is statistically significant.')
else:
    print('The difference in concussion rates on plays with and without returns is not statistically significant.')
print('p-value:', '%f' % ttest_res.pvalue)
plays_with_return = plays_with_punts[plays_with_punts.Has_Return]
plays_with_fair_catch = plays_with_punts[plays_with_punts.Has_Fair_Catch]

label = ['Has_Fair_Catch']

features = [
    'Punt_Distance', 'Time_Passed_Sec', 'Score_Differential',
    'Yard_Line_Absolute', 'Coverage_Distance_1', 'Coverage_Distance_2',
    'Hangtime', 'Temperature', 'Is_Grass', 'Is_Outdoor'
]

plays_with_return_or_fc = pd.concat(
    [plays_with_return, plays_with_fair_catch]
)[['GameKey', 'PlayID'] + features + label].dropna()

x_train, x_test, y_train, y_test = train_test_split(
    np.array(plays_with_return_or_fc[features]),
    np.array(plays_with_return_or_fc[label]).ravel(),
    test_size=0.2,
    random_state=0
)
parameters = {
    'n_estimators':[10, 50, 100],
    'learning_rate': [.01, .05, .1],
    'max_depth': [1, 2, 3]
}

search = GridSearchCV(GradientBoostingClassifier(random_state=0), parameters, cv=3)
search.fit(x_train, y_train)

print(search.best_params_)
# Using best params from grid-search above
gbdt = GradientBoostingClassifier(
    n_estimators=search.best_params_['n_estimators'],
    learning_rate=search.best_params_['learning_rate'],
    max_depth=search.best_params_['max_depth'],
    random_state=0
).fit(x_train, y_train)

y_scores_gbdt = gbdt.predict_proba(x_test)[:,1]

print("Train accuracy: {:,.1%}".format(gbdt.score(x_train, y_train)))
print("Test accuracy: {:,.1%}".format(gbdt.score(x_test, y_test)))
print("ROC AUC: {:,.3f}".format(roc_auc_score(y_test, y_scores_gbdt)))
ax = pd.DataFrame(
    gbdt.feature_importances_,
    index = features,
    columns=['importance']
).sort_values('importance', ascending=True).plot(
    title='Fair-Catch Model Feature Importances', kind='barh', legend=False, figsize=(10,5), fontsize=12
)

ax.title.set_size(16)
logreg = LogisticRegression(solver='liblinear', random_state=2).fit(x_train, y_train)
y_scores_logreg = logreg.predict_proba(x_test)[:,1]
y_pred_logreg = logreg.predict(x_test)

print("Train accuracy:",logreg.score(x_train, y_train))
print("Test accuracy:", logreg.score(x_test, y_test))
print("ROC AUC:", roc_auc_score(y_test, y_scores_logreg))

ax = pd.DataFrame(
    logreg.coef_.transpose(),
    index=features,
    columns=['coefficients']
).sort_values('coefficients').plot(
    kind='barh', legend=False, title='Logistic Regression Feature Coefficients', figsize=(10,5), fontsize=14
)

ax.title.set_size(16)

def update_coverage_distance(row):
    snap_to_punt_time_adjustment = .5 # Percentage of blocking time converted to coverage time
    minimal_coverage_distance = .5 # don't let adjusted coverage distance get smaller than half a yard
    if 'GL' in row.Role_k or 'GR' in row.Role_k:
        return row.Coverage_Distance
    else:
        return max(
            row.Coverage_Distance - (
                row.Yards_Per_Second * (
                    row.Snap_To_Punt_time * snap_to_punt_time_adjustment
                )
            ), minimal_coverage_distance
        )
# Calculate Adjusted Coverage Distances
adjusted_coverage_distances = feather.read_dataframe('../input/nflpuntpreprocessed/adjusted_coverage_distances.feather')

adjusted_coverage_distances['Adj_Coverage_Distance'] = adjusted_coverage_distances.apply(
    update_coverage_distance, axis=1
)

min_acd = adjusted_coverage_distances.loc[
    adjusted_coverage_distances.groupby(['GameKey', 'PlayID'])['Adj_Coverage_Distance'].idxmin()
][['GameKey', 'PlayID', 'Adj_Coverage_Distance', 'Role_k', 'Super_Role_k']]

adjusted_coverage_distances_2 = adjusted_coverage_distances.drop(
    adjusted_coverage_distances.groupby(['GameKey', 'PlayID'])['Adj_Coverage_Distance'].idxmin()
)

second_min_acd = adjusted_coverage_distances_2.loc[
    adjusted_coverage_distances_2.groupby(['GameKey', 'PlayID'])['Adj_Coverage_Distance'].idxmin()
][['GameKey', 'PlayID', 'Adj_Coverage_Distance', 'Role_k', 'Super_Role_k']]
ax = pd.DataFrame({
    "Before Rule Change": min_distances.Super_Role_k.value_counts() / min_acd.shape[0],
    "After Rule Change": min_acd.Super_Role_k.value_counts() / min_acd.shape[0]
}).plot(kind='barh', figsize=(10,5), title='Nearest Coverage Team Member At Time Of Punt Reception')

vals = ax.get_xticks()
ax.set(xlabel="% of Plays", xticklabels=['{:,.0%}'.format(x) for x in vals])

ax.title.set_size(16)
ax.xaxis.label.set_size(14)
# Evaluate trained model on new adjusted coverage distance features
adj_features = [
    'Punt_Distance', 'Time_Passed_Sec', 'Score_Differential',
    'Yard_Line_Absolute', 'Adj_Coverage_Distance_1', 'Adj_Coverage_Distance_2',
    'Hangtime', 'Temperature', 'Is_Grass', 'Is_Outdoor'
]

adj_X = plays_with_return_or_fc.merge(
    min_acd, on=['GameKey', 'PlayID'], how='inner', validate='one_to_one'
).merge(
    second_min_acd, on=['GameKey', 'PlayID'], how='inner', validate='one_to_one', suffixes=["_1", "_2"]
)

adj_X['Adj_Has_Fair_Catch_pred'] = gbdt.predict(adj_X[adj_features]) | adj_X.Has_Fair_Catch
adj_plays_with_punts = plays_with_punts.merge(
    adj_X[['GameKey', 'PlayID', 'Adj_Has_Fair_Catch_pred']],
    on=['GameKey', 'PlayID'], how='left', validate='one_to_one'
)

adj_plays_with_punts['Adj_Punt_Type'] = adj_plays_with_punts.apply(
    lambda row:
        row.Punt_Type
        if pd.isna(row.Adj_Has_Fair_Catch_pred)
        else (
            'fair catch'
            if row.Adj_Has_Fair_Catch_pred
            else 'return'
        )
    , axis=1
)
adj_plays_with_punts['Adj_Has_Fair_Catch_pred'] = adj_plays_with_punts.Adj_Punt_Type == 'fair catch'
adj_plays_with_punts['Adj_Punt_Return_Length'] = adj_plays_with_punts.apply(
    lambda row:
        row.Punt_Return_Length
        if not row.Adj_Has_Fair_Catch_pred
        else 0.0
    , axis=1
)
ax = pd.DataFrame({
    "(Estimated) After Rule Change": adj_plays_with_punts.Adj_Punt_Type.value_counts() / adj_plays_with_punts.shape[0],
    "Before Rule Change": adj_plays_with_punts.Punt_Type.value_counts() / adj_plays_with_punts.shape[0]
}, index=adj_plays_with_punts.Punt_Type.unique()).transpose().plot(
    kind='barh', title='Rule Change Increases Fair Catch Frequency from 25% to 40%', figsize=(10,5)
)
vals = ax.get_xticks()
ax.set(xlabel="% of Plays", xticklabels=['{:,.0%}'.format(x) for x in vals]);

ax.xaxis.label.set_size(14)
ax.title.set_size(16)
pre_concussions = adj_plays_with_punts.Has_Concussion.sum()
post_concussions = (
    adj_plays_with_punts.Has_Concussion & ~(
        (adj_plays_with_punts.Punt_Type == 'return') & (adj_plays_with_punts.Adj_Punt_Type == 'fair catch')
    )
).sum()
print("# Concussions Before Rule Change:", pre_concussions)
print("# Concussions After Rule Change:", post_concussions)
print("Percent Decrease in Concussions: {:,.1%}".format((pre_concussions - post_concussions) / pre_concussions))
adj_plays_with_punts['Next_Poss_Yard_Line'] = adj_plays_with_punts.apply(
    lambda row:
        np.nan
        if row.Punt_Type == 'unreturnable'
        else 100 - (row.Yard_Line_Absolute + row.Punt_Distance - row.Punt_Return_Length)
    , axis=1
)

adj_plays_with_punts['Adj_Next_Poss_Yard_Line'] = adj_plays_with_punts.apply(
   lambda row:
        np.nan
        if row.Adj_Punt_Type == 'unreturnable'
        else 100 - (row.Yard_Line_Absolute + row.Punt_Distance - row.Adj_Punt_Return_Length)
    , axis=1
)

adj_plays_with_returnable_punts = adj_plays_with_punts[
    adj_plays_with_punts.Has_Return | adj_plays_with_punts.Has_Fair_Catch
]

ax = pd.DataFrame(
    {
        'Before Distribution':adj_plays_with_returnable_punts.Next_Poss_Yard_Line,
        'After Distribution': adj_plays_with_returnable_punts.Adj_Next_Poss_Yard_Line
    }
).plot(
    kind='density', xlim=(0,100), ylim=(0,.03), figsize=(12,6),
    title='Distribution of Receiving Team\'s Starting Field Position',
)

before_mean_yard_line = adj_plays_with_returnable_punts.Next_Poss_Yard_Line.mean()
after_mean_yard_line = adj_plays_with_returnable_punts.Adj_Next_Poss_Yard_Line.mean()

yard_line_vals = [0,10,20,30,40,50,60,70,80,90,100]
yard_line_labels = [
    'Own Goal Line' if x == 0
    else (
        'Opp Goal Line' if x == 100
        else (
            x if x <= 50 
            else 100-x
        )
    ) for x in yard_line_vals]
ax.set(
    xlabel="Yard Line", xticks=yard_line_vals, xticklabels=yard_line_labels,
    yticklabels= [''] + [str(x) for x in ax.get_yticks()[1:]]
)
plt.axvline(
    before_mean_yard_line, linestyle='dashed', color='tab:blue', linewidth=2,
    label='Before Mean: {:,.1f} yard line'.format(before_mean_yard_line)
)
plt.axvline(
    after_mean_yard_line, linestyle='dashed', color='tab:orange', linewidth=2,
    label='After Mean: {:,.1f} yard line'.format(after_mean_yard_line)
)
ax.xaxis.label.set_size(14)
ax.title.set_size(16)
plt.legend();
x = plays_with_return[['Punt_Return_Length', 'Coverage_Distance_1']].dropna()
linregress_result = linregress(x.Coverage_Distance_1, x.Punt_Return_Length)
print('R^2: {:,.3f}'.format(linregress_result.rvalue ** 2))
print('P-value: {}'.format(linregress_result.pvalue))
exciting_return_length = 20

print("Of All Punts (returnable and unreturnable)")
print("    Percent of total punt plays with >20 yard return before change: {:,.1%}".format(
    (adj_plays_with_punts.Punt_Return_Length > exciting_return_length).sum() / adj_plays_with_punts.shape[0]
))
print("    Percent of total punt plays with >20 yard return after change: {:,.1%}".format(
    (adj_plays_with_punts.Adj_Punt_Return_Length > exciting_return_length).sum() / adj_plays_with_punts.shape[0]
))