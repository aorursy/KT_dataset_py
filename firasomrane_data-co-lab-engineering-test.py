# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold



import lightgbm as lgb



from fastai import *

from fastai.tabular import *
# lgb.__version__
data = pd.read_csv('/kaggle/input/data co-lab engineering department test/data_co_lab_engineering_dataset.csv')
# data = data.head(2000)

data.head()
data.info()
# These are some categorical features

print(data.nunique())

print(data['Browser'].unique())

# print(data['OperatingSystems'].unique())

# print(data['Region'].unique())

# print(data['TrafficType'].unique())
data.describe()
_ = sns.countplot(data['Revenue'], palette='Set3')
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(4,3,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(4,3,i)

        sns.distplot(df1[feature], hist=False,label=label1)

        sns.distplot(df2[feature], hist=False,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
t0 = data.loc[data['Revenue'] == True]

t1 = data.loc[data['Revenue'] == False]

## keep only numerical columns

cat_columns =  ['Month', 'Browser', 'OperatingSystems', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

con_columns = [col for col in data.columns.values if col not in cat_columns]

plot_feature_distribution(t0, t1, 'True', 'False', con_columns)
correlations = data[con_columns].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

## the least correlated columns

correlations.head(10)
## the most correlated columns

correlations.tail(10)
# from sklearn.decomposition import PCA



# ## Use PCA to reduce the dimensionality : keep only 10 columns

# pca = PCA(n_components=10)

# X = pca.fit_transform(X)

# print 'Shape of X after PCA to the first 10 dimensions: ', X.shape

# Shape of X after PCA to the first 10 dimensions:  (1000, 10)

# In [6]:

# ## PCA can also be used to visualize the data

# plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)

# plt.show()
## determine the list of the columns that contains missing values

missing_cols = data.columns[data.isna().any()].tolist()

## show the dtype of this columns

data[missing_cols].info()
## Imputation techniques: https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779



data_to_impute = data[missing_cols]

imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'

imputed_data = imp_mean.fit_transform(data_to_impute)

data[missing_cols] = pd.DataFrame(imputed_data, columns = missing_cols)

data.info()
## one hot encoding

y = pd.DataFrame()

y['Revenue'] = data['Revenue']

data_copy = data.drop(['Revenue'], axis = 1)

## convert the type of Weekend column since get_dummies converts only columns with object or category dtype

# data_copy['Weekend'] = data_copy['Weekend'].astype(str)

# print(data_copy.info())

X = pd.get_dummies(data_copy, columns = cat_columns)

print(X.shape)

X.columns
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_normalized = pd.DataFrame(scaler.fit_transform(X))

# X_normalized.columns = X.columns

# X_normalized.head()
# label encoding of the target column so we convert False to 0 and True to 1



le = LabelEncoder()

y['Revenue'] = le.fit_transform(y['Revenue'])

y.head()
### Split the data to train and test



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.shape
### Train a logistic regression

lr = LogisticRegression(solver = 'lbfgs')

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



# evaluate the model

## accuracy

print('The accuracy of our logistic regression model is {}'.format(accuracy_score(y_test, y_pred)))

## confusion matrix

cm = confusion_matrix(y_true= y_test, y_pred=y_pred)

cm_df = pd.DataFrame(cm,

                     index = ['False','True'], 

                     columns = ['False','True'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('Logistic regression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
## just used some frequently used parameters for gradient boosting.

param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary',  ## binary classification problem

    'verbosity': 1,

    'is_unbalance ': True,

}
## We use cross validation with 10 folds

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)

oof = np.zeros(len(X_train))

predictions = np.zeros(len(X_test))

features = X_train.columns.values

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y_train.iloc[trn_idx])

    val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y_train.iloc[val_idx])



    num_round = 1000000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)

    oof[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(X_test[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(y_train, oof)))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:10].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,5))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()
y_pred = np.round(predictions)

print('The accuracy of our gradient boosting model is {}'.format(accuracy_score(y_test, y_pred)))

## confusion matrix

cm = confusion_matrix(y_true= y_test, y_pred=y_pred)

cm_df = pd.DataFrame(cm,

                     index = ['False','True'], 

                     columns = ['False','True'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('Gradient Boosting \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
### maybe we have to use a different threshhold other than 0.5

thresh = 0.4

y_pred = np.array([1 if x >thresh else 0 for x in list(predictions)])

print('The accuracy of our gradient boosting model is {}'.format(accuracy_score(y_test, y_pred)))

## confusion matrix

cm = confusion_matrix(y_true= y_test, y_pred=y_pred)

cm_df = pd.DataFrame(cm,

                     index = ['False','True'], 

                     columns = ['False','True'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('Gradient Boosting \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
data_copy = data.copy()

le = LabelEncoder()

data_copy['Revenue'] = le.fit_transform(data_copy['Revenue'])

X_train, X_test, y_train, y_test = train_test_split(data_copy, y, test_size = 0.2, random_state = 0)

X_test.drop(['Revenue'], axis = 1, inplace = True)



## the dependent variable

dep_var = 'Revenue'

## catergorical column names

cat_names = cat_columns

## continuous column names

cont_names = [col for col in con_columns if col != 'Revenue']



## here we specify the preprocessing that must be done

## FillMissing to fill the null values

## Categorify to one hot encode categorical columns

## Normalize which is very important in neural network to get a faster gradient decent.

procs = [FillMissing, Categorify, Normalize]



print("Categorical columns are : ", cat_names)

print('Continuous numerical columns are :', cont_names)
X_train.head()
## construct the test and train TabularList instances

test = TabularList.from_df(X_test, cat_names=cat_names, cont_names=cont_names)



train = (TabularList.from_df(X_train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .random_split_by_pct(valid_pct=0.2, seed=43)

                        .label_from_df(cols=dep_var)

                        .add_test(test, label=0)

                        .databunch())
train.show_batch(rows=10)
learn = tabular_learner(train, layers=[1000,500], metrics=accuracy)
learn.fit_one_cycle(5, 2.5e-2)
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(20, slice(3e-3))
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

pred_prob, pred_class = preds.max(1)
y_pred = pred_class.numpy()

print('The accuracy of our neural network model is {}'.format(accuracy_score(y_test, y_pred)))

## confusion matrix

cm = confusion_matrix(y_true= y_test, y_pred=y_pred)

cm_df = pd.DataFrame(cm,

                     index = ['False','True'], 

                     columns = ['False','True'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('Neural network \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
## version of fastai

print(__version__)