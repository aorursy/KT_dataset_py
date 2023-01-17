import pandas as pd 

import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import make_scorer, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()
test.head()
percent_missing = train.isnull().sum() * 100 / len(train)

missing_vals_df = pd.DataFrame({'Percent Missing': percent_missing})

print(missing_vals_df)
train = train.drop(columns = ['image_name', 'patient_id','benign_malignant','diagnosis'])

train.head()
train.info()
cat_features = ['sex','anatom_site_general_challenge']

print("Categorical features:", cat_features)



num_features = ['age_approx','target']

print("Numerical features:", num_features)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler



class PreprocessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cat_features, num_features):

        self.cat_features = cat_features

        self.num_features = num_features

    

    def fit(self, X, y=None):

        return self

  

    def transform(self, X, y=None): 

        dataframe = X.copy()

        for name in self.cat_features:

            col = pd.Categorical(dataframe[name])

            dataframe[name] = col.codes

    

        # Normalize numerical features

        scaler = MinMaxScaler()

        dataframe[self.num_features] = scaler.fit_transform(dataframe[num_features])

        

        return dataframe
# Preprocessing categorical and numerical features

train_processed = PreprocessTransformer(cat_features, num_features).transform(X = train)



# Imputing missing values 

train_noNan = pd.DataFrame(SimpleImputer().fit_transform(train_processed))

train_noNan.columns = train_processed.columns



train_noNan.head()
percent_missing = train_noNan.isnull().sum() * 100 / len(train_noNan)

missing_vals_df = pd.DataFrame({'Percent Missing': percent_missing})

print(missing_vals_df)
X_train = train_noNan.copy().drop(columns = ['target'])

y_train = train_noNan.copy()['target']



X_train.head()
search_space = [

  {

     'max_depth': [10, 20, 30, 40, 50, 60, None],

     'max_features': ['auto', 'sqrt'],

     'min_samples_leaf': [1, 2, 4],

     'min_samples_split': [2, 5, 10],

     'n_estimators': [200, 400, 600, 800, 1000]

  }

]



cv_method = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)

scoring = {'AUC':make_scorer(roc_auc_score)}
optimizer = RandomizedSearchCV(

  estimator = RandomForestClassifier(),

  param_distributions=search_space,

  cv=cv_method,

  scoring=scoring,

  refit='AUC',

  return_train_score = True,

  verbose=1,

  n_iter = 100,

  n_jobs = 10, 

)



# Approximately 1 hour run time with GPU assistance

rf_model = optimizer.fit(X_train, y_train)
# Display mean AUC score

optimizer.cv_results_['mean_test_AUC'].mean()
# Display most important parameters

optimizer.best_params_
features = X_train.columns

imp_dict = {features[i]:optimizer.best_estimator_.feature_importances_[i] for i in range(len(features))}

imp_dict = sorted(imp_dict.items(), key=lambda x: x[1])

print(imp_dict)



plt.bar(*zip(*imp_dict))

plt.xticks(rotation="vertical")

plt.show()
test.head()
test_copy = test.copy().drop(columns = ['image_name','patient_id'])



cat_features = ['sex','anatom_site_general_challenge']

num_features = ['age_approx']

test_processed = PreprocessTransformer(cat_features,num_features).transform(X = test_copy)



test_noNan = pd.DataFrame(SimpleImputer().fit_transform(test_processed))

test_noNan.columns = test_processed.columns



test_processed.head()
y_pred = rf_model.predict_proba(test_processed[features])

pd.DataFrame(y_pred).head()
y_pred_malignant = [p[1] for p in y_pred]
submission = pd.DataFrame()

submission['image_name'] = test.image_name.values

submission['target'] = y_pred_malignant



submission.to_csv('submission.csv',index = False)

submission.head()