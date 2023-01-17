import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

import pandas as pd
import seaborn as sns
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.automl import H2OAutoML

cc_fraud = pd.read_csv("../input/creditcard.csv")
cc_fraud.head()
cc_fraud.describe()
cc_fraud.Class.value_counts()
# number of zero Amounts
cc_fraud.Amount.loc[cc_fraud.Amount==0].count()
# heatmap only for variables V1 to V28
corr_v = cc_fraud.drop(['Time', 'Amount', 'Class'], axis=1).corr()
sns.heatmap(corr_v, xticklabels=corr_v.columns.values, yticklabels=corr_v.columns.values)
# correlations between Time and Amount and the PCA-variables
corr_sub = cc_fraud.drop(['Class'], axis=1).corr()
corr_sub = corr_sub.loc[~corr_sub.index.isin(['Time', 'Amount']), ['Amount', 'Time']]
corr_sub
corr_sub.loc[(corr_sub.Amount == corr_sub.Amount.max()) | (corr_sub.Amount == corr_sub.Amount.min()), ['Amount']]
corr_sub.loc[(corr_sub.Time == corr_sub.Time.max()) | (corr_sub.Time == corr_sub.Time.min()), ['Time']]
# log transform Amount and box-plot
# In order to make the patterns in the data more interpretable Amount has been log transformed
cc_fraud['Amount_log'] = np.log(cc_fraud.Amount + 0.01)
sns.boxplot(x="Class", y="Amount_log", data=cc_fraud)
plt.show()
cc_fraud.groupby('Class').Amount_log.std()
# we can reject the null hypothesis that fraudulent and non fraudulent amounts have equal variances
stats.levene(cc_fraud.loc[cc_fraud.Class == 0, 'Amount'], cc_fraud.loc[cc_fraud.Class == 1, 'Amount'])
# we are dealing with fairly big sample sizes so a Welch’s t-test with non qual variances should give a non biased result
# the average amount is different across classes
stats.ttest_ind(cc_fraud.loc[cc_fraud.Class == 0, 'Amount'], cc_fraud.loc[cc_fraud.Class == 1, 'Amount'], equal_var=False)
# plot the distributions of the top few variables and Time by class
# look only at the variables that have the highest absolute difference in either mean, median or standard deviation between the
# non fraudulent and the fraudulent class
eda = cc_fraud.drop(['Amount', 'Amount_log', 'Time'], axis=1)\
    .groupby('Class')\
    .aggregate(['mean', 'median', 'std'])\
    .diff()\
    .T\
    .reset_index()
eda['abs_diff'] = eda.iloc[:,3].abs()
eda_sub = eda.sort_values('abs_diff', ascending=False).head(10)
features = np.append( 'Time', eda_sub.level_0.unique())
nplots = np.size(features)
plt.figure(figsize=(15, 4*nplots))
gs = gridspec.GridSpec(nplots, 1)
for i, feat in enumerate(features):
    ax = plt.subplot(gs[i])
    sns.distplot(cc_fraud[feat][cc_fraud.Class == 1], bins=30)
    sns.distplot(cc_fraud.loc[cc_fraud.Class == 0, feat], bins=30)
    ax.legend(['fraudulent', 'non fraudulent'], loc='best')
    ax.set_xlabel('')
    ax.set_title('Feature: ' + feat)
# start up a local 1-node H2O cloud
h2o.init()
# from pandas to h2o
cc_fraud_h2o = h2o.H2OFrame(cc_fraud.drop('Amount_log', axis=1))
# split to train, validation and test
cc_fraud_h2o["Class"] = cc_fraud_h2o["Class"].asfactor()
train, valid, test = cc_fraud_h2o.split_frame([0.7, 0.15], seed=1234)
IV = cc_fraud_h2o.col_names[:-1]
DV = cc_fraud_h2o.col_names[-1]
# Logistic regression
# Time and Amount are on different scales compared to the PCA variables set standardize=True will handle that.
# 28 out of the 30 IV come from a PCA, we dont need to worry about multicollinearity. We can set lambda to be equal to 0 and 
# estimate p_values
glm_binom = H2OGeneralizedLinearEstimator(
    model_id="glm",
    solver="IRLSM",
    family="binomial",
    standardize=True, 
    lambda_=0,
    compute_p_values=True,
    seed=1234
)

glm_binom.train(IV, DV, training_frame=train, validation_frame=valid)
glm_binom.model_performance(test).pr_auc()
# the ROC AUC is indeed very high it will be difficult to distinguish between models using only it
glm_binom.model_performance(test).plot()
glm_binom.model_performance(test).confusion_matrix()
coef_table = glm_binom._model_json['output']['coefficients_table'].as_data_frame()
coef_table['coefficients_abs'] = abs(coef_table['coefficients'])
coef_table.sort_values('coefficients_abs', ascending = False).head()
# Logistic regression with interactions, include only the interactions between the PCA variables and Time and Amount that have
# the highest correlation
# can test further if including mode interactions will improve performance
glm_binom_int = H2OGeneralizedLinearEstimator(
    model_id="glm_int",
    solver="IRLSM",
    family="binomial",
    standardize=True, 
    lambda_=0,
    compute_p_values=True,
    interaction_pairs=[('Time', 'V3'), ('Time', "V5"), ('Amount', 'V2'), ('Amount', 'V7')],
    seed=1234
)

glm_binom_int.train(IV, DV, training_frame=train, validation_frame=valid)
glm_binom_int.model_performance(test).pr_auc()
# Random forest
# min_rows specifies the minimum number of observations for a leaf, its better to be set to a low number since we have an 
# unbalanced sample
rf_mod = H2ORandomForestEstimator(
    model_id="rf",
    binomial_double_trees=False,
    ntrees=200,
    max_depth=8,
    min_rows=25,
    stopping_rounds=2,
    stopping_tolerance=0.001,
    # nfolds=5, # evaluate based on cross validation
    # fold_assignment='stratified' # stratified cross validation
    seed=1234
)
# rf_mod.train(IV, DV, training_frame=train) # when we evaluate based on CV test data AUC is 97.44. It is slower but better
# when we dont have enough data
rf_mod.train(IV, DV, training_frame=train, validation_frame=valid) # evaluate based on the validation data, better test AUC
rf_mod.model_performance(test).pr_auc()
# Random forest with data weighting
# In the data less than 1% of the transactions are fraudulent. If we set balance_class=True the default behaviour is to
# oversample the underrepresented class till a balanced sample is achieved, which means that fraudulent rows will have a weight
# of more than 500. Tested a few less extreme sample weights, both oversampling fraudulent transactions and undersampling
# non fraudulent, and best PR AUC is achieved when giving fraudulent transactions a weight of 5 and non fraudulent a weight 
# of 0.5. Maybe it is worth running a grid search
rf_mod_bal = H2ORandomForestEstimator(
    model_id="rf_bal",
    binomial_double_trees=False,
    ntrees=200,
    max_depth=8,
    min_rows=25,
    stopping_rounds=2,
    stopping_tolerance=0.001,
    balance_classes=True,
    class_sampling_factors=[0.5, 5.0], 
    seed=1234
)
rf_mod_bal.train(IV, DV, training_frame=train, validation_frame=valid)
rf_mod_bal.model_performance(test).pr_auc()
# look at variable importance based on gini
rf_mod_bal.varimp_plot(20)
# partial dependency plot shows us that as V17 increases from -5 to -1, the likelihood of fraudelent transaction decreases
pdp = rf_mod_bal.partial_plot(cols=["V17"], data=train) 
train['V17'].hist(breaks=100)
# Gradient Boosting
gbm = H2OGradientBoostingEstimator(
    model_id="gbm",
    ntrees=200,
    max_depth=8,
    min_rows=25,
    learn_rate=0.01,
    stopping_tolerance=0.001, 
    stopping_rounds=2, # early stopping
    balance_classes=True,
    seed=1234
)
gbm.train(IV, DV, training_frame=train, validation_frame=valid)
gbm.model_performance(test).pr_auc()
# Grid search GBM
gbm_params = {'learn_rate': 0.01,
              'max_depth': [5, 8],
              'min_rows': [15, 25, 35],
              'ntrees': 200,
              'stopping_tolerance': 0.001,
              'stopping_rounds': 2,
              'balance_classes': [True, False],
              'class_sampling_factors': [[0.5, 1.0], [0.5, 5.0], [1.0, 10.0]],
              
              'seed': 1234}
grid_search = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params=gbm_params)
grid_search.train(IV, DV, training_frame=train, validation_frame=valid)
# look at the parameters for the best model
best_prauc = 0
for id_model, model in enumerate(grid_search.models):
    prauc = model.model_performance(test).pr_auc()
    conf_m = model.model_performance(test).confusion_matrix()
    if prauc > best_prauc:
        best_prauc = prauc
        best_conf_m = conf_m
        best_auc_id = id_model
best_model = grid_search[best_auc_id]

max_depth = best_model.params['max_depth']['actual']
min_rows = best_model.params['min_rows']['actual']
sample_rate = best_model.params['sample_rate']['actual']
balance_classes = best_model.params['balance_classes']['actual']
class_sampling_factors = best_model.params['class_sampling_factors']['actual']
print ('''Best model PR AUC: {}, with parameters max_depth: {}, min_rows: {}, sample_rate: {}, balance_classes: {}, 
       class_sampling_factors: {}'''
       .format(best_prauc, max_depth, min_rows, sample_rate, balance_classes, class_sampling_factors))
conf_m
# Forward Neural Net (slow)
dl =  H2ODeepLearningEstimator(
    model_id='dl',
    activation="rectifierwithdropout",
    input_dropout_ratio=0.3,
    hidden=[256, 256, 256],
    stopping_rounds=2,
    stopping_metric="logloss",
    epochs=200,
    seed=1234)
dl.train(IV, DV, training_frame=train, validation_frame=valid)
dl.model_performance(test).pr_auc()
# show me what you can do for 5 mins
aml = H2OAutoML(
    max_runtime_secs=300, 
    exclude_algos=["DeepLearning"],
    seed = 1234)
aml.train(IV, DV, training_frame=train, validation_frame=valid)
aml.leaderboard[['model_id', 'logloss']] # GLM is the top model
# get test AUC for the top model
top_model = aml.leaderboard['model_id'][0, 0]
h2o.get_model(top_model).model_performance(test).pr_auc()
h2o.get_model(top_model).model_performance(test).confusion_matrix()
