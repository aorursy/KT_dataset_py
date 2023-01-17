import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import xgboost as xgb



train_df = pd.read_csv("../input/learn-together/train.csv")

test_df = pd.read_csv("../input/learn-together/test.csv")
train_df.head()
print('Train set size: ', train_df.shape)

print('Test set size: ' , test_df.shape)
print("Missing values in train set: ", train_df.isna().any().any())

print("Missing values in test set: ", test_df.isna().any().any())
train_df.describe()
cat_cols_filter = train_df.columns.str.startswith(('Soil', 'Wild'))

cat_col_names = train_df.loc[:, cat_cols_filter].columns.values



# Iterate through categorcial columns in both train and tests sets to find differences in unique values

# It is also good to know how many unique values are there to help decide what to do in case mismatch is found

for col in cat_col_names:

    if set(train_df[col].unique()) != set(test_df[col].unique()):

        print(f'Col [{col}] value / count:')

        print('-------------- train -----------')

        print(f'{train_df[col].value_counts().to_string()}')

        print('-------------- test ------------')

        print(f'{test_df[col].value_counts().to_string()}\n')

train_df['Cover_Type'].value_counts().plot.bar();
num_col_names = [cname for cname in train_df.columns.values if (cname not in cat_col_names) and (cname != 'Cover_Type')]

corr_matrix_df = train_df[num_col_names].corr()



# Since Pandas do not have a built-in heatmap plot, I'm using pyplot and seaborn here instead

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(corr_matrix_df, annot=True, ax=ax);
train_prep_df = train_df.copy()

test_prep_df = test_df.copy()



train_prep_df.drop(["Id"], axis = 1, inplace=True)

test_ids = test_df["Id"]

test_prep_df.drop(["Id"], axis = 1, inplace=True)



train_prep_df.drop(["Soil_Type7", "Soil_Type15"], axis = 1, inplace=True)

test_prep_df.drop(["Soil_Type7", "Soil_Type15"], axis = 1, inplace=True)



feature_names = [f for f in train_prep_df.columns.values if f != 'Cover_Type']

target_name = 'Cover_Type'



#to make sure I got it right

print(len(feature_names)) #should be 56-3-1 = 52
X_train, X_val, y_train, y_val = train_test_split(train_prep_df[feature_names], train_prep_df[target_name], test_size=0.2, random_state=0)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
rf_model = RandomForestClassifier(n_jobs=4, random_state=0)

rf_model.fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_jobs=4, seed=0)

xgb_model.fit(X_train, y_train)
rf_train_score = rf_model.score(X_train, y_train)

rf_val_score = rf_model.score(X_val, y_val)

print('RF train score: ', rf_train_score)

print('RF val score: ', rf_val_score)



xgb_train_score = xgb_model.score(X_train, y_train)

xgb_val_score = xgb_model.score(X_val, y_val)

print('XGB train score: ', xgb_train_score)

print('XGB val score: ', xgb_val_score)
test_preds_rf = rf_model.predict(test_prep_df)

test_preds_xgb = xgb_model.predict(test_prep_df)



output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_rf})

output.to_csv('initial_rf.csv', index=False);



output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_xgb})

output.to_csv('initial_xgb.csv', index=False);



print('Done!')
# First I will create a new dataframe to hold the feature importance values, so that it is easier to plot them 

feature_df = pd.DataFrame({'feature': feature_names, 'importance': rf_model.feature_importances_})

ax = feature_df.sort_values('importance', ascending=False).plot.bar(x='feature', figsize=(15, 6), fontsize=12)
# What value is the cut-off border? 

print(feature_df[feature_df['feature']=='Soil_Type20'])



#store this value

cutoff = feature_df[feature_df['feature']=='Soil_Type20']['importance'].values[0]

print('\nCut-off val: ', cutoff)
cols_to_keep = feature_df[feature_df['importance']>cutoff]['feature']



# Prepare new set of training / validation data

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_prep_df[cols_to_keep], train_prep_df[target_name], test_size=0.2, random_state=1)

print('New train / val data shape: ', X_train_new.shape, X_val_new.shape, y_train_new.shape, y_val_new.shape)



# Also modify test data set accordingly

test_new_df = test_prep_df[cols_to_keep]

print('New test data shape: ', test_new_df.shape)
rf_model_new = RandomForestClassifier(n_jobs=4, random_state=0);

rf_model_new.fit(X_train_new, y_train_new);



rf_train_score_new = rf_model_new.score(X_train_new, y_train_new)

rf_val_score_new = rf_model_new.score(X_val_new, y_val_new)

print('RF new train score: ', rf_train_score_new)

print('RF new val score: ', rf_val_score_new)

print('RF old val score: ', rf_val_score)



xgb_model_new = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_jobs=4, seed=0)

xgb_model_new.fit(X_train_new, y_train_new);



xgb_train_score_new = xgb_model_new.score(X_train_new, y_train_new)

xgb_val_score_new = xgb_model_new.score(X_val_new, y_val_new)

print('\nXGB new train score: ', xgb_train_score_new)

print('XGB new val score: ', xgb_val_score_new)

print('XGB old val score: ', xgb_val_score)
# check out scikit and xgb docs for more info on which params are best suited for tuning



# RF initial random search through params 

#random_params = {

#    'n_estimators': [100,200,400,800],

#    'max_features': ["sqrt", None, "log2"],

#    'max_depth': [None, 50, 100, 200, 400, 800, 1600],

#    'min_samples_split': [2, 5],

#    'bootstrap': [True, False],

#}

#

# RandomizedSearchCV and GridSearchCV already include cross-validation functionality, and return a model that will be ready for training accordingly

#rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(random_state=0), param_distributions=random_params, scoring='accuracy', n_iter=200, cv=5, n_jobs=4, verbose=10, random_state=0)

#rf_random.fit(X_train_new, y_train_new)

#

# Grid search through reduced list of params

#tune_params = {

#    'n_estimators': [200,400,800],

#    'max_features': ["sqrt", None],

#    'max_depth': [None, 100, 400],

#    'min_samples_split': [2],

#    'bootstrap': [False],

#}

#

#rf_tuned = GridSearchCV(RandomForestClassifier(random_state=0), tune_params, scoring='accuracy', cv=5, scoring='accuracy', n_jobs=4, verbose=10)

#rf_tuned.fit(X_train_new, y_train_new)

# XGBoost grid search

#xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_jobs=4, seed=0)

#

# Grid search through reduced list of params

#random_params = {

#        'n_estimators': [100,200, 400, 800],

#        'min_child_weight': [1, 5, 10],

#        'gamma': [0.5, 1, 1.5, 2, 5],

#        'subsample': [0.6, 0.8, 1.0],

#        'colsample_bytree': [0.6, 0.8, 1.0],

#        'learning_rate': [0.1,0.05,0.001],

#        'max_depth': [3, 4, 5, 10]

#}

#

#xgb_random = RandomizedSearchCV(estimator = xgb_clf, param_distributions=random_params, scoring='accuracy',n_iter=500, cv=5, n_jobs=4, verbose=10, random_state=0)

#xgb_random.fit(X_train_new, y_train_new)
#RF results

#rf_random_val_score = rf_random.score(X_val_new, y_val_new)

#rf_tuned_val_score = rf_random.score(X_val_new, y_val_new)

#

#print("Best parameter set found in random search:")

#print()

#print(rf_random.best_params_)

#print()

#print('RF random val score: ', rf_random_val_score)

#print('RF random val score improvement vs previous: ', rf_random_val_score - rf_val_score_new)

#

#print("\nBest parameter set found in grid search:")

#print()

#print(rf_tuned.best_params_)

#print()

#print('RF tunded val score: ', rf_tuned.score(X_val_new, y_val_new))

#print('RF tuned val score improvement vs previous: ', rf_tuned_val_score - rf_val_score_new)



#XGB results

#xgb_random_val_score = xgb_random.score(X_val_new, y_val_new)

#xgb_tuned_val_score = xgb_random.score(X_val_new, y_val_new)

#

#print("Best parameter set found in random search:")

#print()

#print(xgb_random.best_params_)

#print()

#print('XGB random val score: ', xgb_random_val_score)

#print('XGB random val score improvement vs previous: ', xgb_random_val_score - xgb_val_score_new)

#

#print("\nBest parameter set found in grid search:")

#print()

#print(rf_tuned.best_params_)

#print()

#print('XGB tunded val score: ', xgb_tuned.score(X_val_new, y_val_new))

#print('XGB tuned val score improvement vs previous: ', xgb_tuned_val_score - xgb_val_score_new)
rf_tuned = RandomForestClassifier(bootstrap=False, max_depth=45, max_features='sqrt', min_samples_split=2, min_samples_leaf=1, n_estimators=950, random_state=0, n_jobs=4);

rf_tuned.fit(X_train_new, y_train_new);



rf_tuned_train_score = rf_tuned.score(X_train_new, y_train_new)

rf_tuned_val_score = rf_tuned.score(X_val_new, y_val_new)

print('RF tuned train score: ', rf_tuned_train_score)

print('RF tuned val score: ', rf_tuned_val_score)

print('RF old val score: ', rf_val_score)



xgb_tuned = xgb.XGBClassifier(objective='multi:softmax', num_class=7, n_estimators=500, max_depth=45, subsample=1.0, learning_rate=0.035, min_child_weight=1, gamma=0.5, colsample_bytree=0.8, n_jobs=4, seed=0)

xgb_tuned.fit(X_train_new, y_train_new);



xgb_tuned_train_score = xgb_tuned.score(X_train_new, y_train_new)

xgb_tuned_val_score = xgb_tuned.score(X_val_new, y_val_new)

print('\nXGB tuned train score: ', xgb_tuned_train_score)

print('XGB tuned val score: ', xgb_tuned_val_score)

print('XGB old val score: ', xgb_val_score)
test_preds_rf_tuned = rf_tuned.predict(test_new_df)

test_preds_xgb_tuned = xgb_tuned.predict(test_new_df)



output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_rf_tuned})

output.to_csv('tuned_rf.csv', index=False);



output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_xgb_tuned})

output.to_csv('tuned_xgb.csv', index=False);



print('Done!')
# I scale the whole dataset, and then split into training / validation data again specificaly for SVC training

# Target colkumn is niot scaled

scaler = StandardScaler()

train_scaled_data = scaler.fit_transform(train_prep_df[cols_to_keep])

test_scaled_data = scaler.fit_transform(test_prep_df[cols_to_keep])



train_scaled_df = pd.DataFrame(data=train_scaled_data, columns=cols_to_keep)

test_scaled_df = pd.DataFrame(data=test_scaled_data, columns=cols_to_keep)



train_scaled_df[target_name] = train_prep_df[target_name]



# Prepare new set of training / validation data

X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(train_scaled_df[cols_to_keep], train_scaled_df[target_name], test_size=0.2, random_state=1)

print('Original train / val data shape: ', X_train_new.shape, X_val_new.shape, y_train_new.shape, y_val_new.shape)

print('New scaled train / val data shape: ', X_train_scaled.shape, X_val_scaled.shape, y_train_scaled.shape, y_val_scaled.shape)



print('\nOriginal test data shape: ', test_new_df.shape)

print('New scaled test data shape: ', test_scaled_df.shape)
# Training with the best found params

svc_tuned = SVC(C=130, gamma=0.045, kernel='rbf', decision_function_shape='ovo', random_state=0);

svc_tuned.fit(X_train_scaled, y_train_scaled);



svc_tuned_train_score = svc_tuned.score(X_train_scaled, y_train_scaled)

svc_tuned_val_score = svc_tuned.score(X_val_scaled, y_val_scaled)

print('SVC tuned train score: ', svc_tuned_train_score)

print('SVC tuned val score: ', svc_tuned_val_score)
test_preds_svc_tuned = svc_tuned.predict(test_scaled_df)



output = pd.DataFrame({'Id': test_ids, 'Cover_Type': test_preds_svc_tuned})

output.to_csv('tuned_svc.csv', index=False);



print('Done!')