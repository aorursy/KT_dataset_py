#from https://stackoverflow.com/a/31434967

from IPython.display import display

from IPython.display import HTML

import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)



# This line will hide code by default when the notebook is exported as HTML

di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)



# This line will add a button to toggle visibility of code blocks, for use with the HTML export version

di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import plotly.graph_objs as go

import squarify

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import tools

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')



seed=42
df_raw = pd.read_csv(r'DATA\bank.csv')

df_full = df_raw.copy()
df_full.head()
df_full.shape
df_full.info()
plt.figure(figsize=(16,8))

labels ="Did not deposit", "Deposited"

colors = ['c', 'm']

plt.suptitle('Information on Term Suscriptions', fontsize=20)

plt.pie(df_full["deposit"].value_counts(),autopct='%1.2f%%', labels=labels, colors=colors)

plt.ylabel('% Deposited', fontsize=14)
df_full.hist(bins=20, figsize=(16,8))

plt.show()
df_full = df_full.drop(df_full.loc[df_full.job=="unknown"].index)
df_full.job = df_full.job.replace("admin.", "management")
df_full.job.value_counts()
#need to one-hot encode our target variable

from sklearn.preprocessing import LabelEncoder



fig=plt.figure(figsize=(16,8))

df_full.deposit = LabelEncoder().fit_transform(df_full.deposit)



#we can only use numerical features for correlation analysis

numeric_df_full = df_full.select_dtypes(exclude="object")



cor_diagram = numeric_df_full.corr()



sns.heatmap(cor_diagram, cbar=True)

plt.title("Correlation matrix")

plt.show()
df_full["duration_status"] = np.nan



avg_duration = df_full["duration"].mean()



df_full["duration_status"] = df_full.apply(lambda row: "below_average" if row.duration<avg_duration else "above_average",axis=1 )



yes_no_by_durationstatus = pd.crosstab(df_full['duration_status'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_durationstatus
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_durationstatus.plot(kind='bar', stacked=False)

plt.title("The Impact of Duration vs Opening a Deposit", fontsize=18)

plt.xlabel("Duration Status", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['temp_balance'] = np.ceil(df_full["balance"]/10000)*10000

yes_no_by_temp_balance = pd.crosstab(df_full['temp_balance'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_balance
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_balance.plot(kind='bar', stacked=False)

plt.title("The Impact of Balance vs Opening a Deposit", fontsize=18)

plt.xlabel("Balance", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['temp_age'] = np.ceil(df_full["age"]/10)*10

yes_no_by_temp_age = pd.crosstab(df_full['temp_age'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_age
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_age.plot(kind='bar', stacked=False)

plt.title("The Impact of Age vs Opening a Deposit", fontsize=18)

plt.xlabel("Age", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['temp_pdays_bucket'] = df_full.apply(lambda row: 0 if row['pdays']<=0 else 1 if row['pdays']<=30 else 2 if row['pdays']<=60 else 3 if row['pdays']<=90 else 6 if row['pdays']<=180 else 12 if row['pdays']<=365 else 24 if row['pdays']<=730 else 25,axis=1)
df_full['temp_pdays_bucket'].value_counts()
yes_no_by_temp_pdays = pd.crosstab(df_full['temp_pdays_bucket'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_pdays
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_pdays.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Pdays (months) vs Opening a Deposit", fontsize=18)

plt.xlabel("Pdays (months)", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['campaign'].hist()
df_full['temp_campaign'] = df_full.apply(lambda row: 11 if row['campaign']>=11 else row['campaign'],axis=1)
yes_no_by_temp_campaign = pd.crosstab(df_full['temp_campaign'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_campaign
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_campaign.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Number of times contacted this campaign vs Opening a Deposit", fontsize=18)

plt.xlabel("Number of times contacted", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['poutcome'].hist()
yes_no_by_poutcome = pd.crosstab(df_full['poutcome'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_poutcome
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_poutcome.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of previous campaign vs Opening a Deposit", fontsize=18)

plt.xlabel("Previous campaign outcome", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['previous'].hist()
df_full['temp_previous'] = df_full.apply(lambda row: 11 if row['previous']>=11 else row['previous'],axis=1)
yes_no_by_temp_previous = pd.crosstab(df_full['temp_previous'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_previous
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_previous.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Number of times contacted previous campaign vs Opening a Deposit", fontsize=18)

plt.xlabel("Number of times contacted", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full["duration_status"] = np.nan



avg_duration = df_full["duration"].mean()



df_full["duration_status"] = df_full.apply(lambda row: "below_average" if row.duration<avg_duration else "above_average",axis=1 )



yes_no_by_durationstatus = pd.crosstab(df_full[df_full['poutcome']=="success"]['duration_status'], df_full[df_full['poutcome']=="success"]['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_durationstatus
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_durationstatus.plot(kind='bar', stacked=False)

plt.title("The Impact of Duration vs Opening a Deposit is previous campaign was successful", fontsize=18)

plt.xlabel("Duration Status", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()





df_full["duration_status"] = np.nan



avg_duration = df_full["duration"].mean()



df_full["duration_status"] = df_full.apply(lambda row: "below_average" if row.duration<avg_duration else "above_average",axis=1 )



yes_no_by_durationstatus = pd.crosstab(df_full[df_full['poutcome']=="failure"]['duration_status'], df_full[df_full['poutcome']=="failure"]['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_durationstatus



sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_durationstatus.plot(kind='bar', stacked=False)

plt.title("The Impact of Duration vs Opening a Deposit is previous campaign was a failure", fontsize=18)

plt.xlabel("Duration Status", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['temp_previous'] = df_full.apply(lambda row: 11 if row['previous']>=11 else row['previous'],axis=1)
yes_no_by_temp_previous = pd.crosstab(df_full[df_full['poutcome']=="success"]['temp_previous'], df_full[df_full['poutcome']=="success"]['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_previous
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_previous.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Number of times contacted previous campaign vs Opening a Deposit if previous campaign successful", fontsize=18)

plt.xlabel("Number of times contacted", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
yes_no_by_temp_previous = pd.crosstab(df_full[df_full['poutcome']=="failure"]['temp_previous'], df_full[df_full['poutcome']=="failure"]['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_previous
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_previous.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Number of times contacted previous campaign vs Opening a Deposit if previous campaign not successful", fontsize=18)

plt.xlabel("Number of times contacted", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full['temp_pdays_bucket'] = df_full.apply(lambda row: 0 if row['pdays']<=0 else 1 if row['pdays']<=30 else 2 if row['pdays']<=60 else 3 if row['pdays']<=90 else 6 if row['pdays']<=180 else 12 if row['pdays']<=365 else 24 if row['pdays']<=730 else 25,axis=1)
yes_no_by_temp_pdays = pd.crosstab(df_full[df_full['poutcome']=="success"]['temp_pdays_bucket'], df_full[df_full['poutcome']=="success"]['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_pdays
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_pdays.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Pdays (months) vs Opening a Deposit if previous campaign was successful", fontsize=18)

plt.xlabel("Pdays (months)", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
yes_no_by_temp_pdays = pd.crosstab(df_full[df_full['poutcome']=="failure"]['temp_pdays_bucket'], df_full[df_full['poutcome']=="failure"]['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_pdays
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_pdays.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of Pdays (months) vs Opening a Deposit if previous campaign was unsuccessful", fontsize=18)

plt.xlabel("Pdays (months)", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
yes_no_by_temp_housing = pd.crosstab(df_full['housing'], df_full['deposit']).apply(lambda row: round(row/row.sum(),4) * 100, axis=1)

yes_no_by_temp_housing
sns.set(rc={'figure.figsize':(16,8)})

ax = yes_no_by_temp_housing.plot(kind='bar', stacked=False)#, order=['0m','1m','2m', '3m', '6m','12m', '24m', '>24m'])

plt.title("The Impact of having an existing housing loan vs Opening a Deposit", fontsize=18)

plt.xlabel("Has housing loan", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df_full = df_raw.copy()
df_full.head()
df_full[df_full.age.isnull()]
from sklearn.model_selection import StratifiedShuffleSplit



stratifier = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for training_data_indexes, test_data_indexes in stratifier.split(df_full,df_full.loan):

    stratified_training_data = df_full.loc[training_data_indexes]

    stratified_test_data = df_full.loc[test_data_indexes]

    

    

#create copies

train = stratified_training_data.copy()

test = stratified_test_data.copy()
df_full
train.shape
train
test.shape
from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical or categorical columns

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
train.info()
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()

#numeric_cols.remove('deposit')
numeric_cols
categoric_cols = train.select_dtypes(exclude=np.number).columns.tolist()

#remove target variable, to ensure that we end up with feature set, not label

categoric_cols.remove('deposit')

categoric_cols
from sklearn.preprocessing import OneHotEncoder,StandardScaler
# Build pipelines to scale numerical features and encode categorical



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



# Making pipelines

numerical_pipeline = Pipeline([

    ("select_numeric", DataFrameSelector(numeric_cols)),

    ("std_scaler", StandardScaler()),

])



categorical_pipeline = Pipeline([

    ("select_cat", DataFrameSelector(categoric_cols)),

    ("cat_encoder",  OneHotEncoder(handle_unknown='ignore'))

])



from sklearn.pipeline import FeatureUnion

# https://stackoverflow.com/a/52666039

preprocess_pipeline = FeatureUnion(transformer_list=[

        ("numerical_pipeline", numerical_pipeline),

        ("categorical_pipeline", categorical_pipeline),

    ])
X_train = preprocess_pipeline.fit_transform(train)

X_train = X_train.toarray()

X_train
X_train.shape
categorical_pipeline["cat_encoder"].get_feature_names()
# do not fit the scaler to test features

X_test = preprocess_pipeline.transform(test)

X_test = X_test.toarray()

X_test
#instantiatie label from stratified datasets

y_train = train['deposit']

y_test = test['deposit']

y_train.shape
# encode label

from sklearn.preprocessing import LabelEncoder



encode = LabelEncoder()

y_train = encode.fit_transform(y_train)

y_test = encode.fit_transform(y_test)
feature_names = numeric_cols.copy()

feature_names.extend([col for col in categorical_pipeline["cat_encoder"].get_feature_names()])
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score, cross_val_predict





dict_classifiers = {

    'logistic_reg': LogisticRegression(),

    'support_vector_clf': SVC(),

    'knn_clf': KNeighborsClassifier(),

    'gradient_boosting_clf': GradientBoostingClassifier(),

    'random_forest_cld': RandomForestClassifier(),

    'naive_bayes_clf': GaussianNB(),

    'xgboost_clf': XGBClassifier()

}



dict_preds = {}



for this_classifier in dict_classifiers.keys():

    this_score = cross_val_score(dict_classifiers[this_classifier], X_train, y_train, cv=5)

    # https://stackoverflow.com/questions/25006369/what-is-sklearn-cross-validation-cross-val-score

    this_mean = this_score.mean()

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html

    dict_preds[this_classifier] = cross_val_predict(dict_classifiers[this_classifier], X_train, y_train, cv=5) #get predictions, for ROC curve

    print(this_classifier, this_mean)
from sklearn.metrics import confusion_matrix

#https://machinelearningmastery.com/confusion-matrix-machine-learning/

 

for this_classifier in dict_preds.keys():

    this_pred = dict_preds[this_classifier]

    confusion_results = confusion_matrix(y_train, this_pred)

    print(this_classifier, "\n\n", confusion_results, "\n\n")
from sklearn.metrics import roc_auc_score



for this_classifier in dict_preds.keys():

    print('{}: {}'.format(this_classifier, roc_auc_score(y_train, dict_preds[this_classifier])))
import xgboost as xgb
# https://stackoverflow.com/a/46943417

dtrain = xgb.DMatrix(X_train, label=y_train,feature_names=feature_names)

dtest = xgb.DMatrix(X_test, label=y_test,feature_names=feature_names)
from sklearn.metrics import mean_absolute_error
# https://xgboost.readthedocs.io/en/latest/parameter.html

params = {

    # Parameters that we are going to tune.

    'max_depth':6,

    'min_child_weight': 1,

    'eta':.3,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'reg:linear',

}



params['eval_metric'] = "auc"

params['verbosity'] = 0



num_boost_round = 999
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10

)
cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=42,

    nfold=5,

    metrics={'auc'},

    early_stopping_rounds=10

)



cv_results
gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(6,9)

    for min_child_weight in range(1,8)

]
# Define initial best params and AUC

# min_mae = float("Inf") # use this if for MAE

max_auc = 0

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

    # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'auc'},

        early_stopping_rounds=10

    )

    # Update best AUC

    mean_auc = cv_results['test-auc-mean'].max() 

    # mean_rmse = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-auc-mean'].argmax()

    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))

    if mean_auc > max_auc:

        max_auc = mean_auc

        best_params = (max_depth,min_child_weight)

print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
params['max_depth'] = 6

params['min_child_weight'] = 3
params
gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(1,11)]

    for colsample in [i/10. for i in range(1,11)]

]
max_auc = 0

best_params = None

# We start by the largest values and go down to the smallest

for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))

    # We update our parameters

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'auc'},

        early_stopping_rounds=10

    )

    # Update best AUC

    mean_auc = cv_results['test-auc-mean'].max() 

    # mean_rmse = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-auc-mean'].argmax()

    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))

    if mean_auc > max_auc:

        max_auc = mean_auc

        best_params = (subsample,colsample)

print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
params['subsample'] = 0.9

params['colsample_bytree'] = 0.8
%time

# This can take some time…



min_mae = float("Inf")

best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:

    print("CV with eta={}".format(eta))

    # We update our parameters

    params['eta'] = eta

    # Run and time CV

    %time

    

    # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'auc'},

        early_stopping_rounds=10

    )

    # Update best AUC

    mean_auc = cv_results['test-auc-mean'].max() 

    # mean_rmse = cv_results['test-mae-mean'].min()

    boost_rounds = cv_results['test-auc-mean'].argmax()

    print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))

    if mean_auc > max_auc:

        max_auc = mean_auc

        best_params = eta

print("Best params: {}, AUC: {}".format(best_params, max_auc))
params['eta'] = 0.01

params
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10

)
num_boost_round = model.best_iteration + 1

num_boost_round
our_best_model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")]

)
params
XGBClassifier(num_boost_round=num_boost_round)
dict_classifiers['xgboost_clf'] = XGBClassifier(colsample_bytree=0.8, eta=0.01,

                      eval_metric='auc', max_depth=6,

                      min_child_weight=3, objective='reg:linear',

                      subsample=0.9,seed=42,num_boost_round=804,verbosity=0)



dict_classifiers
# accuracy

revised_score = cross_val_score(dict_classifiers['xgboost_clf'], X_train, y_train, cv=5)



revised_score.mean()
revised_preds = cross_val_predict(dict_classifiers['xgboost_clf'], X_train, y_train, cv=5)
confusion_results = confusion_matrix(y_train, revised_preds)

confusion_results
revised_auc = roc_auc_score(y_train, revised_preds)

revised_auc
our_best_model.save_model("my_model.model")
our_best_predictions = our_best_model.predict(dtest)

final_best_predictions = [abs(round(value)) for value in our_best_predictions]
confusion_results = confusion_matrix(y_test, final_best_predictions)

confusion_results
revised_auc = roc_auc_score(y_test, final_best_predictions)

revised_auc
from sklearn.metrics import accuracy_score



accuracy_score(y_test, final_best_predictions)
from xgboost import plot_importance
plot_importance(our_best_model)

plt.show()
import statsmodels.api as sm
model = sm.GLM.from_formula("deposit ~ duration + balance + age + pdays + campaign + poutcome + previous + housing ", family=sm.families.Binomial(), data=train)

result = model.fit()

result.summary()
dummy_df = pd.DataFrame(train[['duration', 'previous', 'poutcome']])

dummy_df['poutcome'] = dummy_df.apply(lambda row: 1 if row['poutcome']=='success' else 0,axis=1)

dummy_df_cor = dummy_df.corr()

pd.DataFrame(np.linalg.inv(dummy_df.corr().values), index = dummy_df_cor.index, columns=dummy_df_cor.columns)