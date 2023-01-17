import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.model_selection import train_test_split

import category_encoders as ce

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import GroupShuffleSplit

from sklearn.ensemble import GradientBoostingRegressor 

from mlxtend.regressor import StackingCVRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

import time

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import StackingRegressor



df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

df.head(3)
train_inds, val_inds = next(GroupShuffleSplit(test_size=.35, n_splits=2, random_state = 42).split(df, groups=df['Patient']))

train = df.iloc[train_inds]

val = df.iloc[val_inds]



col = 'Patient'



cardinality = len(pd.Index(df[col]).value_counts())

print("Number of " + df[col].name + "s in original DataFrame df: " + str(cardinality) + '\n')   

cardinality = len(pd.Index(train[col]).value_counts())

print("Number of " + train[col].name + "s in train: " + str(cardinality) + '\n')

cardinality = len(pd.Index(val[col]).value_counts())

print("Number of " + val[col].name + "s in val: " + str(cardinality))



target = 'FVC'

features = train.drop(columns=[target, 'Patient']).columns.tolist()



X_train = train[features]

y_train = train[target]

X_val = val[features]

y_val = val[target]
X_train.head(1)
%%time



numeric_features = ['Age', 'Percent', 'Weeks']

numeric_transformer = Pipeline(steps=[

    ('scaler', StandardScaler())])



categorical_features = ['SmokingStatus', 'Sex']

categorical_transformer = Pipeline(steps=[

    ('ordinal', OrdinalEncoder())])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



pipe = Pipeline([

    ('preprocessor', preprocessor),

    ('select', SelectKBest()),

    ('model', HistGradientBoostingRegressor(random_state=0))])



param_grid = {

    'select__k': [1, 2, 3, 4],

    'model__max_depth': [2, 4, 6, 8, 10, 12],

    'model__min_samples_leaf': [1, 10, 20, 40, 50, 80, 100]}



search = GridSearchCV(

            pipe, 

            param_grid, 

            verbose=10,

            cv=5)

search.fit(X_train, y_train)
results_df = pd.DataFrame(search.cv_results_).sort_values('mean_test_score', ascending=False)

results_df.head(3)
results_df['params'][0]
pipelineHGBR = make_pipeline(preprocessor, 

                         HistGradientBoostingRegressor(random_state=0,

                                                      max_depth=2,

                                                      min_samples_leaf=1))



pipelineHGBR.fit(X_train, y_train)

y_train_pred = pipelineHGBR.predict(X_train)

y_val_pred = pipelineHGBR.predict(X_val)



print("Train r2 score",r2_score(y_train,y_train_pred))

print("Val r2 score",r2_score(y_val,y_val_pred))

sns.distplot(y_val, label='Actual')

sns.distplot(y_val_pred, label='Validation Prediction')

plt.legend();
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test = pd.merge(sub, test, on = ['Patient'], how='outer')

test.drop(columns = ['Patient_Week', 'Patient','FVC_x','Confidence','Weeks_y','FVC_y'], inplace=True)

test.rename(columns = {'Weeks_x':'Weeks'}, inplace = True)



test.head(3)
s = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')



y_test_pred = pipelineHGBR.predict(test)

sub = s[['Patient_Week']]

sub['FVC'] = y_test_pred

sub['Confidence'] = sub['FVC'].std()

sub['FVC'] = sub['FVC'].astype(int)

sub['Confidence'] = sub['Confidence'].astype(int)

sub.head(3)
sub.to_csv("submission_HistGradientBoostingRegressor.csv", index=False)

a = pd.read_csv('./submission_HistGradientBoostingRegressor.csv')

a.head(2)
%% time

#########

ridge = Ridge()

lasso = Lasso()

rf = RandomForestRegressor()



stack = StackingCVRegressor(regressors=(lasso, ridge),

                            meta_regressor=rf, 

                            use_features_in_secondary=True)



pipeline = make_pipeline(preprocessor, 

                         stack)



params = {

        'stackingcvregressor__lasso__alpha': [0.1, 1.0, 7.0],# regularization, wherein we penalize the number of features in a model in order to only keep the most important features. the higher the alpha, the most feature coefficients are zero.

        'stackingcvregressor__ridge__alpha': [0.1, 1.0, 7.0],

        'stackingcvregressor__meta_regressor__max_depth': [2, 4, 6, 8, 10, 12],

        'stackingcvregressor__meta_regressor__min_samples_leaf': [1, 10, 20, 40, 50, 80, 100, 150, 200]

         }



searchR = GridSearchCV(

    verbose=10,

    estimator=pipeline, 

    param_grid=params, 

    cv=5,

    refit=True

)



# estimator=pipeline

# estimator.get_params()



searchR.fit(X_train, y_train)
results_dfR = pd.DataFrame(searchR.cv_results_).sort_values('mean_test_score', ascending=False)

results_dfR.head(3)
results_dfR['params'][0]
params = results_dfR['params'][0]



ridge = Ridge(alpha = 0.1)

lasso = Lasso(alpha = 0.1)

rf = RandomForestRegressor(

                            max_depth = 2,

                            min_samples_leaf = 1,

                            random_state = 0

                        )



estimators = [

    ('lrr', ridge),

    ('lrl', lasso)

]



stack = StackingRegressor(estimators=estimators,

                            final_estimator=rf)



pipelineRfr = make_pipeline(preprocessor, 

                         stack)



pipelineRfr.fit(X_train, y_train)

y_train_pred = pipelineRfr.predict(X_train)

y_val_pred = pipelineRfr.predict(X_val)



print("Train r2 score",r2_score(y_train,y_train_pred))

print("Val r2 score",r2_score(y_val,y_val_pred))

sns.distplot(y_val, label='Actual')

sns.distplot(y_val_pred, label='Validation Prediction')

plt.legend();
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test = pd.merge(sub, test, on = ['Patient'], how='outer')

test.drop(columns = ['Patient_Week', 'Patient','FVC_x','Confidence','Weeks_y','FVC_y'], inplace=True)

test.rename(columns = {'Weeks_x':'Weeks'}, inplace = True)



test.head(3)
s = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')



y_test_pred = pipelineRfr.predict(test)

sub = s[['Patient_Week']]

sub['FVC'] = y_test_pred

sub['Confidence'] = sub['FVC'].std()

sub['FVC'] = sub['FVC'].astype(int)

sub['Confidence'] = sub['Confidence'].astype(int)

sub.head(3)
sub.to_csv("submission_RandomForestRegressor.csv", index=False)

sub.to_csv("submission.csv", index=False)

a = pd.read_csv('./submission_RandomForestRegressor.csv')

a.head(2)
params = {'n_estimators': 500,

          'max_depth': 4,

          'min_samples_split': 5,

          'learning_rate': 0.01,

          'loss': 'ls'}



model = GradientBoostingRegressor(**params)

pipelineGBR = make_pipeline(preprocessor, 

                         model)

pipelineGBR.fit(X_train, y_train)

r2 = r2_score(y_val, pipelineGBR.predict(X_val))

print("The r2_score on test set: {:.4f}".format(r2))
print("Val r2 score",r2_score(y_val,y_val_pred))

sns.distplot(y_val, label='Actual')

sns.distplot(y_val_pred, label='Validation Prediction')

plt.legend();
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test = pd.merge(sub, test, on = ['Patient'], how='outer')

test.drop(columns = ['Patient_Week', 'Patient','FVC_x','Confidence','Weeks_y','FVC_y'], inplace=True)

test.rename(columns = {'Weeks_x':'Weeks'}, inplace = True)



test.head(3)
y_pred = pipelineGBR.predict(test)

sub = s[['Patient_Week']]

sub['FVC'] = y_pred

sub['Confidence'] = sub['FVC'].std()

sub['FVC'] = sub['FVC'].astype(int)

sub['Confidence'] = sub['Confidence'].astype(int)

sub.head(2)
sub.to_csv("submission_GradientBoostingRegressor.csv", index=False)

a = pd.read_csv('./submission_GradientBoostingRegressor.csv')

a.sample(10)