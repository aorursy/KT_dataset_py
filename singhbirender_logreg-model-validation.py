import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import statsmodels.formula.api as sm

import scipy.stats as stats

import pandas_profiling



%matplotlib inline

plt.rcParams['figure.figsize'] = 10, 7.5

plt.rcParams['axes.grid'] = True

plt.gray()



from matplotlib.backends.backend_pdf import PdfPages



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices
bankloans = pd.read_csv("../input/bankloans.csv")
bankloans.columns
bankloans.info()
# pandas_profiling.ProfileReport(bankloans)
numeric_var_names = [key for key, val in bankloans.dtypes.items() if val in ["float64","int64","float32","int32"] ]

cat_var_names = [key for key, val in bankloans.dtypes.items() if val in ["object"]]

print(numeric_var_names)

print(cat_var_names)
bankloans_num = bankloans[numeric_var_names]

bankloans_num.head()
bankloans_cat = bankloans[cat_var_names]

bankloans_cat.head()
def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_summary=bankloans_num.apply(lambda x: var_summary(x)).T
num_summary
bankloans_existing = bankloans_num[bankloans_num.default.isnull()==0]

bankloans_new = bankloans_num[bankloans_num.default.isnull() == 1]
def outlier_capping(x):

    x = x.clip_upper(x.quantile(0.99))

    x = x.clip_lower(x.quantile(0.01))

    return x

bankloans_existing = bankloans_existing.apply(lambda x: outlier_capping(x))
bankloans_existing.corr()
sns.heatmap(bankloans_existing.corr())

plt.show()
bankloans_existing.columns.difference(["default"])
bp = PdfPages("Transformation plots.pdf")



for num_variable in bankloans_existing.columns.difference(['default']):

    binned = pd.cut(bankloans_existing[num_variable], bins = 10, labels = list(range(1,11)))

    odds = bankloans_existing.groupby(binned)["default"].sum()/(bankloans_existing.groupby(binned)["default"].count() - bankloans_existing.groupby(binned)["default"].sum())

    log_odds = np.log(odds)

    fig, axes = plt.subplots(figsize= (10,4))

    sns.barplot(x = log_odds.index, y = log_odds)

    plt.title(str("Logit plot for identifying if the bucketing is required or not for varibale -- ")+ str(num_variable))

    bp.savefig(fig)

bp.close()
bankloans_existing.columns
logreg_model = sm.logit("default~address", data = bankloans_existing).fit()
p = logreg_model.predict(bankloans_existing)
metrics.roc_auc_score(bankloans_existing["default"], p)
2*metrics.roc_auc_score(bankloans_existing["default"], p)-1
somersd_df = pd.DataFrame()

for num_variable in bankloans_existing.columns.difference(["default"]):

    logreg = sm.logit(formula = "default~"+str(num_variable), data = bankloans_existing)

    result = logreg.fit()

    y_score = pd.DataFrame(result.predict())

    y_score.columns = ["Score"]

    somers_d = 2* metrics.roc_auc_score(bankloans_existing["default"], y_score)-1

    temp = pd.DataFrame([num_variable, somers_d]).T

    temp.columns = ["Variable Name", "SomersD"]

    somersd_df = pd.concat([somersd_df, temp], axis = 0)

somersd_df



    
somersd_df.sort_values(by = "SomersD", ascending = False)
X = pd.concat([bankloans_existing[bankloans_existing.columns.difference(["default"])], bankloans_existing["default"]], axis = 1)

features = "+".join(bankloans_existing.columns.difference(["default"]))

X.head()
features
a, b = dmatrices(formula_like = "default~" + features, data = X, return_type = "dataframe")

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]

vif["features"] = b.columns

print(vif)
train, test = train_test_split(bankloans_existing, test_size = 0.3, random_state = 43)

train.columns
logreg = sm.logit(formula = "default ~ address+age+creddebt+debtinc+employ", data = train)

result = logreg.fit()
print(result.summary2())
train_gini = 2* metrics.roc_auc_score(train["default"], result.predict(train))-1

print("The Gini Index for the model built on the Train Data is : ", train_gini)



test_gini = 2* metrics.roc_auc_score(test["default"], result.predict(test))-1

print("The Gini Index for the model built on the Test Data is :", test_gini)
train_auc = metrics.roc_auc_score(train["default"], result.predict(train))

test_auc = metrics.roc_auc_score(test["default"], result.predict(test))



print("The AUC for the model built on the Train Data is:", train_auc)

print("The AUC for the model built on the Test Data is:", test_auc)
train_predicted_prob = pd.DataFrame(result.predict(train))

train_predicted_prob.columns = ["prob"]

train_actual = train["default"]
train_predict=  pd.concat([train_actual, train_predicted_prob], axis = 1)

train_predict.columns = ["actual", "prob"]

train_predict.head()
test_predicted_prob = pd.DataFrame(result.predict(test))

test_predicted_prob.columns = ["prob"]

test_actual = test["default"]
test_predict = pd.concat([test_actual, test_predicted_prob], axis = 1)

test_predict.columns = ["actual", "prob"]

test_predict.head()
np.linspace(0,1,50)
train_predict.head()

train_predict["predicted"] = train_predict["prob"].apply(lambda x: 0.0 if x<0.2 else 1.0)
train_predict.head()
train_predict['tp'] = train_predict.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==1 else 0.0, axis=1)

train_predict['fp'] = train_predict.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==1 else 0.0, axis=1)

train_predict['tn'] = train_predict.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==0 else 0.0, axis=1)

train_predict['fn'] = train_predict.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==0 else 0.0, axis=1)
train_predict.head()
accuracy = (train_predict.tp.sum()+train_predict.tn.sum())/(train_predict.tp.sum()+train_predict.tn.sum()+train_predict.fp.sum()+train_predict.fn.sum())

accuracy
Sensitivity = (train_predict.tp.sum())/(train_predict.tp.sum()+train_predict.fn.sum())

Sensitivity
roc_like_df = pd.DataFrame()

train_temp = train_predict.copy()

for cut_off in np.linspace(0,1,50):

    train_temp['cut_off'] = cut_off

    train_temp['predicted'] = train_temp['prob'].apply(lambda x: 0.0 if x < cut_off else 1.0)

    train_temp['tp'] = train_temp.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==1 else 0.0, axis=1)

    train_temp['fp'] = train_temp.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==1 else 0.0, axis=1)

    train_temp['tn'] = train_temp.apply(lambda x: 1.0 if x['actual']==0.0 and x['predicted']==0 else 0.0, axis=1)

    train_temp['fn'] = train_temp.apply(lambda x: 1.0 if x['actual']==1.0 and x['predicted']==0 else 0.0, axis=1)

    sensitivity = train_temp['tp'].sum() / (train_temp['tp'].sum() + train_temp['fn'].sum())

    specificity = train_temp['tn'].sum() / (train_temp['tn'].sum() + train_temp['fp'].sum())

    accuracy = (train_temp['tp'].sum()  + train_temp['tn'].sum() ) / (train_temp['tp'].sum() + train_temp['fn'].sum() + train_temp['tn'].sum() + train_temp['fp'].sum())

    roc_like_table = pd.DataFrame([cut_off, sensitivity, specificity, accuracy]).T

    roc_like_table.columns = ['cutoff', 'sensitivity', 'specificity', 'accuracy']

    roc_like_df = pd.concat([roc_like_df, roc_like_table], axis=0)

roc_like_df.head()
roc_like_df["total"] = roc_like_df["sensitivity"] + roc_like_df["specificity"]

roc_like_df.head()
roc_like_df[roc_like_df["total"] == roc_like_df["total"].max()] 
roc_like_df[roc_like_df['sensitivity']==roc_like_df['sensitivity'].max()]
test_predict['predicted'] = test_predict['prob'].apply(lambda x: 1 if x > 0.2 else 0)

train_predict['predicted'] = train_predict['prob'].apply(lambda x: 1 if x > 0.2 else 0)
pd.crosstab(train_predict['actual'], train_predict['predicted'])
pd.crosstab(test_predict['actual'], test_predict['predicted'])
print("The overall accuracy score for the Train Data is : ", metrics.accuracy_score(train_predict.actual, train_predict.predicted))

print("The overall accuracy score for the Test Data  is : ", metrics.accuracy_score(test_predict.actual, test_predict.predicted))
print(metrics.classification_report(train_predict.actual, train_predict.predicted))
print(metrics.classification_report(test_predict.actual, test_predict.predicted))
train_predict['Deciles']=pd.qcut(train_predict['prob'],10, labels=False)

train_predict.head()
test_predict['Deciles']=pd.qcut(test_predict['prob'],10, labels=False)

test_predict.head()
no_1s = train_predict[['Deciles','actual']].groupby(train_predict.Deciles).sum().sort_index(ascending=False)['actual']

no_total = train_predict[['Deciles','actual']].groupby(train_predict.Deciles).count().sort_index(ascending=False)['actual']

max_prob = train_predict[['Deciles','prob']].groupby(train_predict.Deciles).max().sort_index(ascending=False)['prob']

min_prob = train_predict[['Deciles','prob']].groupby(train_predict.Deciles).min().sort_index(ascending=False)['prob']
Decile_analysis_train = pd.concat([ min_prob, max_prob, no_1s, no_total-no_1s, no_total], axis=1)

Decile_analysis_train.columns = ['Min_prob', 'Max_prob', '#1', '#0', 'Total']

Decile_analysis_train
no_1s = test_predict[['Deciles','actual']].groupby(test_predict.Deciles).sum().sort_index(ascending=False)['actual']

no_total = test_predict[['Deciles','actual']].groupby(test_predict.Deciles).count().sort_index(ascending=False)['actual']

max_prob = test_predict[['Deciles','prob']].groupby(test_predict.Deciles).max().sort_index(ascending=False)['prob']

min_prob = test_predict[['Deciles','prob']].groupby(test_predict.Deciles).min().sort_index(ascending=False)['prob']



Decile_analysis_test = pd.concat([min_prob, max_prob, no_1s, no_total-no_1s, no_total], axis=1)



Decile_analysis_test.columns = ['Min_prob', 'Max_prob', '#1', '#0', 'Total']



Decile_analysis_test
bankloans_new.head()
bankloans_new['prob'] = result.predict(bankloans_new)
bankloans_new.head()
bankloans_new['default'] = bankloans_new['prob'].apply(lambda x: 1 if x > 0.20 else 0)
bankloans_new.head()
bankloans_new.default.value_counts()