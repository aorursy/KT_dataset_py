# standard
import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option("display.max_columns",None)
df = pd.read_csv('../input/bouts_out_new.csv')
df.head()
fil = ((df.height_A < 224) & (df.height_A > 147) &
      (df.height_B < 224) & (df.height_B > 147) &
      (df.weight_B > 70) & (df.weight_A > 70) &
      (df.age_A < 60) & (df.age_A > 14) &
      (df.age_B < 60) & (df.age_B > 14) &
      (df.reach_A < 250) & (df.reach_A > 130) &
      (df.reach_B < 250) & (df.reach_B > 130)) 
df = df[fil]
df['Diff_age'] = df.age_A - df.age_B
df['Diff_weight'] = df.weight_A - df.weight_B
df['Diff_height'] = df.height_A - df.height_B
df['Diff_reach'] = df.reach_A - df.reach_B

df['Tot_fight_A'] = df.won_A + df.lost_A + df.drawn_A
df['Tot_fight_B'] = df.won_B + df.lost_B + df.drawn_B
df['Diff_exp'] = df.Tot_fight_A - df.Tot_fight_B

df['Win_per_A'] = df.won_A / df.Tot_fight_A
df.loc[df.Tot_fight_A == 0, 'Win_per_A'] = 0 #because maybe it is the first fight
df['Win_per_B'] = df.won_B / df.Tot_fight_B
df.loc[df.Tot_fight_B == 0, 'Win_per_B'] = 0
df['KO_perc_A'] = df.kos_A / df.won_A
df.loc[df.won_A == 0, 'KO_perc_A'] = 0
df['KO_perc_B'] = df.kos_B / df.won_B
df.loc[df.won_B == 0, 'KO_perc_B'] = 0

df.loc[df.stance_A == df.stance_B, 'Stance'] = 0
df.loc[(df.stance_A == 'orthodox') & (df.stance_B == 'southpaw'), 'Stance'] = 1
df.loc[(df.stance_B == 'orthodox') & (df.stance_A == 'southpaw'), 'Stance'] = -1
# A number of these columns don't seem to make a difference now I have 
# differences between boxer A and boxer B
ml = df.copy()
try:
    ml.drop(['judge1_A', 'judge1_B', 'judge2_A', 'judge2_B', 'judge3_A', 'judge3_B', 'decision'], axis=1, inplace=True)
    ml.drop(['age_A','age_B', 'height_A','height_B','reach_A', 'reach_B','stance_A', 'stance_B'], axis=1, inplace=True)
    ml.drop(['weight_A', 'weight_B','won_A','won_B','lost_A','lost_B','drawn_A','drawn_B','Stance'], axis=1, inplace=True) 
except:
    print('already dropped judges')
    
ml.head()
ml_dummies = pd.get_dummies(ml)
ml_dummies.fillna(value=0, inplace=True)
ml_dummies.head()
# Remove the label from the dataframe
try:
    label = ml_dummies['result_win_A']
    del ml_dummies['result_win_A']
    del ml_dummies['result_win_B']
    del ml_dummies['result_draw']
except:
    print("label already removed")
ml_dummies.head()
# import random column
ml_dummies['---randomColumn---'] = np.random.randint(0,1000, size=len(ml_dummies))
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(ml_dummies, label, test_size=0.3)

# Classifiers
from sklearn.ensemble import RandomForestClassifier

classifiers = [
    RandomForestClassifier(),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=17, max_features=6, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=True)    
]

# iterate over classifiers
for item in classifiers:
    classifier_name = ((str(item)[:(str(item).find("("))]))
    print (classifier_name)
    
    # Create classifier, train it and test it.
    clf = item
    clf.fit(feature_train, label_train)
    score = clf.score(feature_test, label_test)
    print (round(score,3),"\n", "- - - - - ", "\n")
    
importance_df = pd.DataFrame()
importance_df['feature'] = ml_dummies.columns
importance_df['importance'] = clf.feature_importances_    

# importance_df.sort_values('importance', ascending=False)
importance_df.set_index(keys='feature').sort_values(by='importance', ascending=True).plot(kind='barh', figsize=(20, 15))
from sklearn.model_selection import GridSearchCV

max_depth_range = range(2,20,5)
leaf_range = range(1,5,2)
n_estimators_range = range(10,140,20)
max_features_range = range(1,len(ml_dummies.columns),5)


param_grid = dict(max_depth = max_depth_range,
                 min_samples_leaf = leaf_range,
                 n_estimators = n_estimators_range,
                 max_features = max_features_range
                )


### Warning, can take some time
# d_tree = RandomForestClassifier()
# grid = GridSearchCV(d_tree, param_grid, cv=5, scoring = 'accuracy', verbose=1, return_train_score=True)
# grid.fit(feature_train, label_train)
# print (grid.best_score_)
# print (grid.best_params_)
# print (grid.best_estimator_)
corr_table = ml_dummies.copy()
corr_table['result_win_A'] = label
corr_table.corr()
fig = plt.figure(figsize = (20,10))
corr = corr_table.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
