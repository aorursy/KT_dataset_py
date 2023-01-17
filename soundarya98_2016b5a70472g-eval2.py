import pandas as pd

import numpy as np

import sklearn as skl

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 20))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
y=df['class']

features=['chem_1','chem_2','chem_4', 'chem_6','attribute']

x= df[features]

x
seed = 7

test_size = 0.3

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=2000)

# import lightgbm 

# from lightgbm import LGBMClassifier

# model = LGBMClassifier(objective='multiclass',  n_estimators=100, num_leaves=20)



# from xgboost import XGBClassifier

# model = XGBClassifier(max_depth=10, learning_rate=0.2, n_estimators=2000, silent=True, objective='multi:softprob', booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None)

# from sklearn.ensemble import ExtraTreesClassifier

# model = ExtraTreesClassifier(n_estimators=1500)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred=np.round(y_pred).astype(int)

# calculate accuracy

from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))
# from xgboost import XGBClassifier

# model = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=100, silent=True, objective='multi:softprob', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None)

# from sklearn.ensemble import ExtraTreesClassifier

# model = ExtraTreesClassifier()



# from lightgbm import LGBMClassifier

# model = LGBMClassifier(objective='multiclass', random_state=5)

# from sklearn.ensemble import ExtraTreesRegressor

# model = ExtraTreesRegressor(n_estimators=2000)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=2000)



model.fit(x, y)



actualtestx=pd.read_csv('test.csv')

x_test = actualtestx[features]

# # make predictions for test data

y_pred_submissions = model.predict(x_test)

y_pred_submissions=np.round(y_pred_submissions).astype(int)
y_pred_submissions=pd.DataFrame(data=y_pred_submissions)



answer=pd.concat([actualtestx['id'], y_pred_submissions], axis=1)



answer.columns=['id', 'class']

answer.to_csv('lab2_9(1).csv', index=False)
dfshuffle=df.iloc[np.random.permutation(len(df))]

dff=dfshuffle.reset_index(drop=True)

dff.head()
dff_class = dff['class'].values
from sklearn.cross_validation import train_test_split

training_indices, validation_indices = training_indices, testing_indices = train_test_split(dff.index,

                                                                                            stratify = dff_class,

                                                                                            train_size=0.8, test_size=0.2)
from tpot import TPOTClassifier

from tpot import TPOTRegressor



tpot = TPOTClassifier(generations=5,verbosity=2)



tpot.fit(dff.drop('class',axis=1).loc[training_indices].values,

         dff.loc[training_indices,'class'].values)
tpot.score(dff.drop('class',axis=1).loc[validation_indices].values,

           dff.loc[validation_indices, 'class'].values)
params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(x, y), verbose=3, random_state=1001 )

random_search.fit(x, y)