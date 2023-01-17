#!pip install pandas, numpy, matplotlib, seaborn, sklearn, xgboost, lightgbm, catboost
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.linear_model     import LogisticRegression

from sklearn.neighbors        import KNeighborsClassifier

from sklearn.svm              import SVC

from sklearn.tree             import DecisionTreeClassifier

from sklearn.ensemble         import RandomForestClassifier

from sklearn.ensemble         import ExtraTreesClassifier

from sklearn.ensemble         import GradientBoostingClassifier

from xgboost                  import XGBClassifier, plot_tree

import lightgbm as lgb



from sklearn.preprocessing    import LabelEncoder

from sklearn.model_selection  import train_test_split

from sklearn.model_selection  import StratifiedKFold

from sklearn.model_selection  import cross_val_score

from sklearn.model_selection  import GridSearchCV

from sklearn.metrics          import accuracy_score, balanced_accuracy_score



import catboost

from catboost import *

from catboost import datasets

from catboost import CatBoostClassifier
df = pd.read_csv("beer_train_clean.csv")

df.head()
cat = df.select_dtypes(include=[object]).columns

print("\nCategorical features:\n", cat.values)
style_encoder = LabelEncoder()

style_encoder.fit(df['Style'])



df[cat] = df[cat].apply(LabelEncoder().fit_transform)

#df['Style'] = df['Style'].apply(style_encoder.fit_transform)

#df['Style'] = style_encoder.fit_transform(df['Style'])

df.head()
from sklearn.impute import SimpleImputer



# Imputar

imputed_df = pd.DataFrame(SimpleImputer().fit_transform(df))

# Restaurar nombres de columnas

imputed_df.columns = df.columns

df = imputed_df

df.head()
# Features

X = df[['Size(L)', 'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime',

        'BoilGravity', 'Efficiency', 'MashThickness', 'PitchRate',

        'PrimaryTemp', 'SugarScale_Plato', 'SugarScale_Specific Gravity',

        'BrewMethod_All Grain', 'BrewMethod_BIAB', 'BrewMethod_Partial Mash', 'BrewMethod_extract'

       ]]



# Label

y = df["Style"].values

X[0:5]
# Train and test split (used only for KNN)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4)

print ('Train:', X_train.shape,  y_train.shape)

print ('Valid:', X_valid.shape,  y_valid.shape)
# KNN

def get_best_k(draw_plot):

  Ks = 20 # max number of Ks

  k = 1 # best K

  mean = np.zeros((Ks-1)) # empty array to store the mean accuracy

  std = np.zeros((Ks-1))  # empty array for standard deviation



  for n in range(1,Ks):

      knn = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train) # train model

      KNN_yhat = knn.predict(X_valid) # get a predicted value from the valid set

      mean[n-1] = accuracy_score(y_valid, KNN_yhat) # add the accuracy to the mean array



      std[n-1] = np.std(KNN_yhat==y_valid)/np.sqrt(KNN_yhat.shape[0]) # add the standard deviation to the std array

  

  k = mean.argmax()+1



  # Plot

  if not draw_plot:

    return k

  plt.plot(range(1, Ks),mean, 'g') # plot the line

  plt.fill_between(range(1, Ks), mean - std, mean + std, alpha=0.10) # fill the standard deviation zone

  plt.legend(('Accuracy ', '+/- std'))

  plt.ylabel('Accuracy ')

  plt.xlabel('Number of Neighbors (K)')

  plt.tight_layout()

  plt.show()

  print("Peak accuracy with", mean.max(), "k =", k)

  

  return k
models = [

    ('Logistic Regression', LogisticRegression(n_jobs=-1)),

    ('SVM',                 SVC()),

    ('Decision Tree',       DecisionTreeClassifier()),

    ('KNN',                 KNeighborsClassifier(n_neighbors=get_best_k(False))),

    ('Extra Trees',         ExtraTreesClassifier(n_jobs=-1)),

    ('Random Forest',       RandomForestClassifier(n_jobs=-1)),

    ('Gradient Boosting',   GradientBoostingClassifier()),

    ('XGBoost',             XGBClassifier(n_estimators=250)),

    ('CatBoost',            CatBoostClassifier(CatBoostClassifier(n_estimators=350, learning_rate=0.1, early_stopping_rounds=10)))

]
Modelnames = []

outcome = []



for name, model in models:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    cv_r = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    outcome.append(cv_r)

    Modelnames.append(name)

    print("%s: %.2f%% (%.2f%%)" % (name, cv_r.mean()*100, cv_r.std()*100))
models = [None]*2

outcome = []

Modelnames = [None]*2
# CatBoost

models[-1] = ('CatBoost',            CatBoostClassifier(iterations=350,learning_rate=0.05, verbose=0))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

cv_r = cross_val_score(models[-1][1], X, y, cv=skf, scoring='accuracy')

outcome.append(cv_r)

Modelnames[-1] = models[-1][0]

print("%s: %.2f%% (%.2f%%)" % (models[-1][0], cv_r.mean()*100, cv_r.std()*100))
# XGBoost better

models[-2] = ('XGBoost', XGBClassifier(n_estimators=250,max_depth=3,learning_rate=0.1,early_stopping_rounds=10))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

cv_r = cross_val_score(models[-2][1], X, y, cv=skf, scoring='accuracy')

outcome.append(cv_r)

Modelnames[-2] = models[-2][0]

print("%s: %.2f%% (%.2f%%)" % (models[-2][0], cv_r.mean()*100, cv_r.std()*100))
# xgboost grid search

xgb_model = XGBClassifier()

optimization_dict = {

        'early_stopping_rounds': [10,15],

        'n_estimators':[250,300,500],

        'max_depth': [3,4],

        'learning_rate': [0.01,0.1]

        }

jobs = 15



model = GridSearchCV(xgb_model, optimization_dict, 

                     scoring='accuracy',pre_dispatch=jobs*2,n_jobs=jobs,verbose=5)

model.fit(X,y)

print(model.best_score_)

print(model.best_params_)



with open('best.txt', 'w+') as f:

    f.write(str(model.best_score_))

    f.write('\n')

    f.write(str(model.best_params_))
# catBoost grid search

cb_model = CatBoostClassifier() # 0.6423168245605418

optimization_dict = {

        'early_stopping_rounds': [10], # 10

        'n_estimators': [350,500],# 350

        'learning_rate': [0.1], # 0.1

        #'l2_leaf_reg': [2, 4, 6],

        #'one_hot_max_size': [50],

        #'min_child_weight': [1, 5, 10],

        #'gamma': [0.5, 1, 2, 5],

        #'subsample': [0.6, 0.8, 1.0],

        'max_depth': [3],# 3

        'random_seed': [0]

        }

#0.6399067544096877 ???

#{'early_stopping_rounds': 10, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 400}

#0.641402582035515

#{'early_stopping_rounds': 10, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}

#0.6402807501598418

#{'early_stopping_rounds': 10, 'l2_leaf_reg': 6, 'learning_rate': 0.15, 'max_depth': 3, 'n_estimators': 400, 'one_hot_max_size': 50, 'random_seed': 0}

jobs = 20



model = GridSearchCV(cb_model, optimization_dict, cv=3,

                     scoring='accuracy',pre_dispatch=2*jobs,n_jobs=jobs,verbose=0)

model.fit(X,y)

print(model.best_score_)

print(model.best_params_)



with open('best.txt', 'w+') as f:

    f.write(str(model.best_score_))

    f.write('\n')

    f.write(str(model.best_params_))
param = {'num_leaves': 31, 'objective': 'multiclassova'}

#param['metric'] = 'auc'

num_round = 10

bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
results = {'Names': Modelnames, 'Results': outcome}

plt.figure(figsize=(16,6))

sns.boxplot(x='Names', y='Results', data=results)
def encode_and_bind(original_dataframe, feature_to_encode):

    dummies = pd.get_dummies(original_dataframe[feature_to_encode], prefix=feature_to_encode)

    res = pd.concat([original_dataframe, dummies], axis=1)

    res = res.drop([feature_to_encode], axis=1)

    return(res)
# Import test set

test_df = pd.read_csv('beer_test.csv')

# Separate categorical values

test_cat = test_df.select_dtypes(include=[object]).columns

# Encoding

test_df[test_cat] = test_df[test_cat].apply(LabelEncoder().fit_transform)

# One Hot Encoding

for c in test_cat.values:

    test_df = encode_and_bind(test_df, c)

# Imputing

imputed_test_df = pd.DataFrame(SimpleImputer().fit_transform(test_df))

imputed_test_df.columns = test_df.columns

test_df = imputed_test_df



test_df.head()
# Test Features

'''

X_test = test_df[['Size(L)', 'OG', 'FG', 'ABV',	'IBU', 'Color', 'BoilSize',

        'BoilTime', 'BoilGravity', 'Efficiency', 'MashThickness',

        'SugarScale', 'BrewMethod', 'PitchRate', 'PrimaryTemp'

       ]]

'''

X_test = test_df.drop('Id', axis=1)

X_test[0:5]
# Predict values from the test set

#preds = XGBClassifier(n_estimators=300, max_depth=3).fit(X, y).predict(X_test)

#preds = CatBoostClassifier(n_estimators=350, max_depth=3, learning_rate=0.1, early_stopping_rounds=10).fit(X, y).predict(X_test)



preds = model.predict(X_test)
from sklearn.inspection      import permutation_importance

# Compute Permutation Feature Importance

pfi = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=0, n_jobs=-1)



# Clean data

sorted_idx = pfi.importances_mean.argsort()[::-1]

pfi_df = pd.DataFrame(data=pfi.importances[sorted_idx].T, columns=X_valid.columns[sorted_idx])



# Plot (This can be barplot, boxplot, violinplot,...)

plt.figure(figsize=(12,4))

sns.barplot(data=pfi_df, orient="h").set_title("Permutation Feature Importance (validation set)",  fontsize=20);
# Generate solution Dataframe

final_df = pd.DataFrame(preds, columns = ['Style'])

# Name Index to 'Id'

final_df.index.name = 'Id'

final_df.head()
# Decode values

final_df['Style'] = style_encoder.inverse_transform(final_df['Style'].astype('int'))

final_df.head()
final_df.shape
# Save to file

final_df.to_csv("Desafío cervezas.csv")