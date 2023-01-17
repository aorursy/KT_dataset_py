import os

import random

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np

import pandas as pd



from lightgbm import LGBMClassifier



#xtreme gradient boosting

from xgboost import XGBClassifier



#neural net

from sklearn.neural_network import MLPClassifier



#HistGradientBoostingClassifier

# explicitly require this experimental feature

from sklearn.experimental import enable_hist_gradient_boosting

#stacking CV classifier

from mlxtend.classifier import StackingCVClassifier



from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



random_state = 1

random.seed(random_state)

np.random.seed(random_state)





print('> Loading data')

X_train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')



y_train = X_train.pop('Cover_Type').astype('int8')





print('> Processing features')

# - https://www.kaggle.com/jakelj/basic-ensemble-model

# - https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575

# - https://www.kaggle.com/kwabenantim/forest-cover-feature-engineering



# Join train and test

X = pd.concat([X_train, X_test])



# Add new features

X['Hydro_Elevation_diff'] = (X['Elevation'] - 

                             X['Vertical_Distance_To_Hydrology'])



X['Hydro_Fire_sum'] = (X['Horizontal_Distance_To_Hydrology'] + 

                       X['Horizontal_Distance_To_Fire_Points'])



X['Hydro_Fire_diff'] = (X['Horizontal_Distance_To_Hydrology'] - 

                        X['Horizontal_Distance_To_Fire_Points']).abs()



X['Hydro_Road_sum'] = (X['Horizontal_Distance_To_Hydrology'] +

                       X['Horizontal_Distance_To_Roadways'])



X['Hydro_Road_diff'] = (X['Horizontal_Distance_To_Hydrology'] -

                        X['Horizontal_Distance_To_Roadways']).abs()



X['Road_Fire_sum'] = (X['Horizontal_Distance_To_Roadways'] + 

                      X['Horizontal_Distance_To_Fire_Points'])



X['Road_Fire_diff'] = (X['Horizontal_Distance_To_Roadways'] - 

                       X['Horizontal_Distance_To_Fire_Points']).abs()



X['Soil_Type'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))



soil_count = X['Soil_Type'].value_counts().to_dict()

X['Soil_count'] = X['Soil_Type'].map(soil_count)



soil_elevation = X.groupby('Soil_Type')['Elevation'].median().to_dict()

X['Soil_Elevation'] = X['Soil_Type'].map(soil_elevation)



# Drop features not useful for classification

drop_cols = ['Aspect', 'Slope',  'Hillshade_3pm', 'Soil_Type']

drop_cols += ['Soil_Type{}'.format(i) for i in range(1, 41)]

drop_cols = [col for col in drop_cols if col in X.columns]



X = X.drop(drop_cols, axis='columns')



# Drop features with low variance in training set

lo_var_cols = []

max_mode_freq = len(X_train) - 10



for col in X.columns:

    if  X.loc[X_train.index, col].value_counts().iat[0] > max_mode_freq:

        lo_var_cols.append(col)



if lo_var_cols:

    X = X.drop(lo_var_cols, axis='columns')



# Scale and bin features

X.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(X))

X = X.astype('int8')



# Separate train and test

X_train = X.loc[X_train.index, :]

X_test = X.loc[X_test.index, :]



del X

print('> No. of features = {}'.format(X_train.columns.size))
print('> Setting up classifiers')

max_features = min(30, X_train.columns.size)



ab_clf = AdaBoostClassifier(n_estimators=200,

                            base_estimator=DecisionTreeClassifier(

                                min_samples_leaf=2,

                                random_state=random_state),

                            random_state=random_state)



et_clf = ExtraTreesClassifier(n_estimators=300,

                              min_samples_leaf=2,

                              min_samples_split=2,

                              max_depth=50,

                              max_features=max_features,

                              random_state=random_state,

                              n_jobs=-1)



lg_clf = LGBMClassifier(n_estimators=300,

                        num_leaves=128,

                        verbose=-1,

                        random_state=random_state,

                        n_jobs=-1)



rf_clf = RandomForestClassifier(n_estimators=300,

                                random_state=random_state,

                                n_jobs=1)



xgb_clf = XGBClassifier(max_depth=10, learning_rate=0.02, n_estimators=1000, verbosity=1,

                         objective='multi:softmax', booster='gbtree', 

                         tree_method='auto', n_jobs=1, gamma=0, 

                         min_child_weight=1, max_delta_step=0, subsample=1,

                         colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,

                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

                         base_score=0.5, random_state=0, missing=None,

                         num_parallel_tree=1, importance_type='gain')



#nn_clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

 #            beta_2=0.999, early_stopping=True, epsilon=1e-08,

  #           hidden_layer_sizes=(500,500), learning_rate='constant',

   #          learning_rate_init=0.001, max_iter=500, momentum=0.9,

    #         n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,

     #        random_state=None, shuffle=True, solver='adam', tol=0.0001,

      #       validation_fraction=0.1, verbose=False, warm_start=False)



hg_clf = HistGradientBoostingClassifier(loss="auto", learning_rate=0.1, max_iter=100,

                                        max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,

                                        l2_regularization=0.0, max_bins=256, scoring=None,

                                        validation_fraction=0.1, n_iter_no_change=None,

                                        tol=1e-07, verbose=0, random_state=None)



ensemble = [("HistGradClassifier", hg_clf),

            ("XGBClassifier", xgb_clf),

            ('AdaBoostClassifier', ab_clf),

            ('ExtraTreesClassifier', et_clf),

            ('LGBMClassifier', lg_clf),

            ('RandomForestClassifier', rf_clf)]







for label, clf in ensemble:

    print ("> Cross-validating classifier ", label)

    score = cross_val_score(clf, X_train.values, y_train.values,

                            cv=5,

                            scoring='accuracy',

                            verbose=1,

                            n_jobs=1)



    print('  -- {: <24} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))
print('> Fitting stack')

# - https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python



clflist = []

for m in ensemble: clflist.append (m[1])

stack = StackingCVClassifier(classifiers=clflist,

                             meta_classifier=rf_clf,

                             cv=5,

                             stratify=True,

                             shuffle=True,

                             use_probas=True,

                             use_features_in_secondary=True,

                             verbose=0,

                             random_state=random_state,

                             n_jobs=1)



stack = stack.fit(X_train.values, y_train.values)


print('> Making predictions')

predictions = stack.predict(X_test.values)
print('> Creating submission')

predictions = pd.Series(predictions, index=X_test.index, dtype=y_train.dtype)

predictions.to_csv('submission.csv', header=['Cover_Type'], index=True, index_label='Id')



print('> Done !')