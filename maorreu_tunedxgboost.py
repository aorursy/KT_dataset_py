# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn')

import xgboost as xgb

from sklearn.ensemble import *

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

from sklearn.tree import *

from sklearn.pipeline import Pipeline

from sklearn.model_selection import *

from sklearn.linear_model import *

from sklearn.preprocessing import *

from sklearn.svm import *

from sklearn.metrics import *

from sklearn.neural_network import *

from sklearn.feature_selection import *

from sklearn.decomposition import *

from sklearn.manifold import *

import catboost 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_path = '/kaggle/input/bgutreatmentoutcome/'

train_df = pd.read_csv(os.path.join(base_path, 'train.CSV'))

test_df = pd.read_csv(os.path.join(base_path, 'test.CSV'))
print(f'train shape: {train_df.shape}')

print(f'test shape: {test_df.shape}')
train_df.head()
total_missing_values = train_df.isnull().sum().sum()

total_values = train_df.shape[0] * train_df.shape[1]

print(f'the fraction of missing values from data: {total_missing_values / total_values}')
missing_val_per_col = train_df.isnull().sum().sort_values() / train_df.shape[0]

missing_val_per_col.plot(figsize=(10, 6), x_compat=True)
num_df = train_df[missing_val_per_col.index].select_dtypes('number')

num_df.columns = pd.CategoricalIndex(num_df.columns, ordered=False)

num_df.hist(figsize=(26, 20), bins=50)

plt.show()
cat_df = train_df[missing_val_per_col.index].select_dtypes('object')

cols = 10

fig, axes = plt.subplots(round(len(cat_df.columns) / cols), cols, figsize=(22, 20))



for i, ax in enumerate(fig.axes):

    if i < len(cat_df.columns):

        sns.countplot(x=cat_df.columns[i], data=cat_df, ax=ax)



fig.tight_layout()
all_data = pd.concat([train_df, test_df])

categorical_df = test_df.select_dtypes(['object'])

all_data_imp = pd.get_dummies(all_data, columns=categorical_df.columns)



y_train = train_df['CLASS']

X_train = all_data_imp[all_data_imp['CLASS'].notnull()].drop(columns=['CLASS']).select_dtypes(['number'])



labels = LabelEncoder().fit_transform(y_train.values)

dtrain = xgb.DMatrix(X_train, label=labels)





X_test = all_data_imp[all_data_imp['CLASS'].isnull()].drop(columns=['CLASS']).select_dtypes(['number'])

dtest = xgb.DMatrix(X_test)



print(all_data.shape)

print(X_train.shape, train_df.shape)

print(X_test.shape, test_df.shape)

# X_train.select_dtypes(['object'])
%%time

imp = SimpleImputer(strategy='mean')

skb = SelectKBest(f_classif, k=80)

pca = SparsePCA(n_jobs=-1, verbose=0, max_iter=200, alpha=8)



################################################################



selected_X_train_filed = X_train.fillna(X_train.mean()).fillna(0)

selected_X_train = skb.fit_transform(selected_X_train_filed, labels)



cols = skb.get_support(indices=True)

cols_names = selected_X_train_filed.iloc[:,cols].columns



selected_X_train = pd.DataFrame(selected_X_train, columns=cols_names)

selected_X_train = pd.DataFrame(pca.fit_transform(selected_X_train, labels), columns=selected_X_train.columns)



###################################################################



selected_X_test = X_test.fillna(X_test.mean()).fillna(0)

selected_X_test = pd.DataFrame(skb.transform(selected_X_test), columns=cols_names)

selected_X_test = pd.DataFrame(pca.transform(selected_X_test), columns=selected_X_test.columns)





selected_dtrain = xgb.DMatrix(selected_X_train, label=labels)

selected_dtest = xgb.DMatrix(selected_X_test)
clf = xgb.XGBClassifier(seed=1301,verbose_eval=True, n_jobs =-1, max_depth =6, learning_rate=0.2, reg_lambda =200, tree_method='gpu_hist')
xgb_param = clf.get_xgb_params()

print ('Start cross validation')

cvresult = xgb.cv(xgb_param, selected_dtrain, metrics=['auc'],num_boost_round=2000, early_stopping_rounds=50, stratified=True, seed=1301, verbose_eval=True, shuffle=True, nfold=4)

cvresult
xgb_param = clf.get_xgb_params()

bst = xgb.train(xgb_param, selected_dtrain, num_boost_round=268, verbose_eval=True)
bst.save_model('0004.model')
# # Load model

# bst = xgb.Booster({'nthread': 4})  # init model

# bst.load_model('0001.model')  # load data
ypred = bst.predict(selected_dtest)

ypred
submition_columns = ['Id', 'ProbToYes']

submition_df = pd.DataFrame()

submition_df['Id'] = X_test.index + 1

submition_df['ProbToYes'] = bst.predict(selected_dtest)
submition_df.to_csv('submition4.csv', index=False)
xgb_param = clf.get_xgb_params()



X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(selected_X_train, labels, test_size=0.2, random_state=42)

selected_dtrain_s = xgb.DMatrix(X_train_s, label=y_train_s)    

selected_dtest_s = xgb.DMatrix(X_test_s) 

bst_s = xgb.train(xgb_param, selected_dtrain_s, num_boost_round=268, verbose_eval=True)
def plot_auc_curve(bst, selected_dtest, y_test):

    ypred = bst.predict(selected_dtest)

    

    print(f'AUC: {roc_auc_score(y_test, ypred)}')

    fpr, tpr, thresholds = roc_curve(y_test, ypred)

    

    plt.plot(fpr,tpr)

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.show()
plot_auc_curve(bst_s, selected_dtest_s, y_test_s)
X_test_s = pd.DataFrame(X_test_s, columns=selected_X_train.columns)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(bst_s)



row_to_show = 10
data_for_prediction = X_test_s.iloc[row_to_show]

# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0], data_for_prediction)
data_for_prediction = X_test_s.iloc[20]

# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0], data_for_prediction)
data_for_prediction = X_test_s.iloc[100]

# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0], data_for_prediction)
shap_values = explainer.shap_values(X_test_s)

shap.summary_plot(shap_values, X_test_s ,max_display=5)
best_features = ['A107', 'A17_O', 'A109', 'A44_O', 'A106']

shap_values = explainer.shap_values(X_test_s, y_test_s)

for feature in best_features:

    shap.dependence_plot(feature, shap_values, X_test_s, interaction_index=None)