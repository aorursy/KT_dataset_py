# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/datmin-joints-2020/train_data.csv")

test_df = pd.read_csv("../input/datmin-joints-2020/test_data.csv")
train_df['zee'] = 'train'

test_df['zee'] = 'test'



combined_df = pd.concat((train_df, test_df))



combined_df = combined_df.sort_values('id').reset_index(drop=True)

combined_df = combined_df.set_index('id')



combined_df.sample(8, random_state=1)
import re



def clean_numeric(df):

    for col in df.columns:

        for cell in df[col]:

            try:

                float(cell)

            except:

                if (all(c.isdecimal() == False for c in cell)):

                    df[col].loc[(df[col] == cell)] = np.nan

                    print(col, cell, "alpha")

                else:

                    df[col].loc[(df[col] == cell)] = int(re.sub("\D", "", cell))

                    print(col, cell, "num---convert:", re.sub("\D", "", cell))
check = combined_df[combined_df.columns[~combined_df.columns.isin(['zee'])]]

clean_numeric(check)
for index, row in check.iterrows():

    for col in check.columns:

        if float(row[col]) < 0:

            print(row[col], index)

check.replace([-1,-9],[np.nan, np.nan], inplace=True)
features = check[check.columns[~check.columns.isin(['Result'])]]

# nan_cols = [i for i in features.columns if features[i].isnull().any()]

# for col in nan_cols:

#     features[col].fillna(features[col].median(), inplace=True)

# features = features.dropna(axis=0)

# features = features.astype('float64')

features.shape
x = features.isnull().sum(axis=1)

x[x > 1.3].loc[:3620].sort_values(ascending=False)[:10]
features = features.drop([1800,1792, 2814])
features.describe()
features.shape
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

# imp = KNNImputer(n_neighbors=2, weights="uniform")

imp = SimpleImputer(missing_values=np.nan, strategy='median')

features_filled = pd.DataFrame(imp.fit_transform(features))

features_filled.columns = features.columns

features_filled.index = features.index
features_filled.shape
features_filled.describe()
features = features_filled.copy()
ax = sns.boxplot(x="word-38",data=features_filled, dodge=False)

ax
# slicing only train that will be tranformed

features_train = features.loc[:3620].copy()

features_train
# take non outlier by threshold Z-score: 3

from scipy import stats

features_train = features_train.loc[(np.abs(stats.zscore(features_train)) < 3).all(axis=1)]
features_train
features_test = features.loc[3621:]

features_test
ax = sns.boxplot(x="word-1",data=features_train, dodge=False)

ax
# features = check[check.columns[~check.columns.isin(['Result'])]]

# nan_cols = [i for i in features.columns if combined_df.query('zee == "train"')[i].isnull().any()]

# for col in nan_cols:

#     combined_df.query('zee == "train"')[col].fillna(features[col].median(), inplace=True)
features_train.shape
features = features_train.append(features_test)

features
combined_df
result_zee = combined_df[['Result', 'zee']]

result_zee
combined_df = features.merge(result_zee, on='id')

combined_df
# col = ['word-' + str(num) + '_y' for num in range(1, 41)] + ['Result', 'zee']

# combined_df = combined_df[col]

# combined_df.columns = ['word-' + str(num) for num in range(1, 41)] + ['Result', 'zee']

# print('shape', combined_df.shape)

# combined_df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

fig=plt.gcf()

fig.set_size_inches(20, 10)

sns.heatmap(features.corr(), cbar=True)
spam_df = combined_df.query('Result == 1').drop([ 'zee'], axis=1)

spam_df.head()
nonspam_df = combined_df.query('Result == 0').drop(['zee'], axis=1)

nonspam_df.head()
import seaborn as sns

import matplotlib.pyplot as plt

fig=plt.gcf()

fig.set_size_inches(20, 10)

sns.heatmap(spam_df.corr(), cbar=True)
import seaborn as sns

import matplotlib.pyplot as plt

fig=plt.gcf()

fig.set_size_inches(20, 10)

sns.heatmap(nonspam_df.corr(), cbar=True)
# corr with result

df = combined_df.drop(['zee'], axis=1)

fig=plt.gcf()

fig.set_size_inches(20, 10)

sns.heatmap(df.corr(), cbar=True)
# CLEANED DATA HERE

# imp_feat = [4,8,11,14,15,25,27,29,33,34,35]

# imp_feat = [33, 34]

imp_feat = []

features = ['word-' + str(num) for num in imp_feat]

X = combined_df.drop(['Result', 'zee'] + features, axis=1)

y = combined_df['Result']

result_zee = combined_df[['Result', 'zee']]

X.head()
from sklearn.preprocessing import PolynomialFeatures

intr_features = ['word-34', 'word-18']

intr_df = X[intr_features]

X.drop(intr_features, axis=1, inplace=True)



pf = PolynomialFeatures(degree=2, interaction_only=False,  

                        include_bias=False)

res = pf.fit_transform(intr_df)

res
X_interaction = pd.DataFrame(res,index=X.index, columns=['word-34', 'word-18',  

                                           '34^2', 

                                           '34 x 18',  

                                           '18^2'])

X_interaction.index
X_all = X.merge(X_interaction, on='id')

X = X_all
# from sklearn.preprocessing import PolynomialFeatures

# intr_features = ['word-15', 'word-33']

# intr_df = X[intr_features]

# X.drop(intr_features, axis=1, inplace=True)



# pf = PolynomialFeatures(degree=2, interaction_only=False,  

#                         include_bias=False)

# res = pf.fit_transform(intr_df)

# X_interaction = pd.DataFrame(res,index=X.index, columns=['word-15', 'word-33',  

#                                            '15^2', 

#                                            '15 x 33',  

#                                            '33^2'])

# X = X.merge(X_interaction, on='id')
# X.describe()
X.columns
'''

0  - 9 = 1

10 - 19 = 2

20 - 29 = 3

30 - 39 = 4

40 - 49 = 5

50 - 59 = 6

'''

# for col in X.columns:

#     X[col + '_bin'] = np.array(np.floor(np.array(X[col]) / 4.))

# col = [c for c in X.columns if 'bin' not in c]

# X = X.drop(col, axis=1)
import matplotlib.pyplot as plt

def show_hist(X, col, fig, ax):

    mu = X[col].mean()  # mean of distribution

    sigma = X[col].std()  # standard deviation of distribution

    num_bins = 50



    # fig, ax = plt.subplots()



    # the histogram of the data

    n, bins, patches = ax.hist(X[col], num_bins, density=1)



    # add a 'best fit' line

    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

         np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    ax.plot(bins, y, '--')

    ax.set_xlabel(col)

    ax.set_ylabel('Probability density')

    ax.set_title(r'Histogram of %s: $\mu=%f$, $\sigma=%d$' % (col, mu, sigma))



    # Tweak spacing to prevent clipping of ylabel

    fig.tight_layout()

# fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(20, 20))



# count = 1

# for i in range(4):

#     for j in range(10):

#         show_hist(X, 'word-' + str(count),fig, axes[j][i])

#         count+=1

# plt.show()
# transform positive

# X_transform = X.transform(lambda x: (1+x)/2)

# X_Transform.sample(100)
# X_Transform.eq(0).any().any()
# X_Transform = X_Transform.drop(['word-34', 'word-33'], axis=1)
# from scipy import stats

# copy = pd.DataFrame({})

# for col in X.columns:

#     data = X_Transform[X_Transform > 0][col].values

#     xt, _ = stats.boxcox(data)

#     copy[col] = xt
# mu = xt.mean()  # mean of distribution

# sigma = xt.std()  # standard deviation of distribution

# num_bins = 50



# fig, ax = plt.subplots()



# # the histogram of the data

# n, bins, patches = ax.hist(xt, num_bins, density=1)



# # add a 'best fit' line

# y0 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

# ax.plot(bins, y0, '--')

# ax.set_xlabel('word-1')

# ax.set_ylabel('Probability density')

# ax.set_title(r'Histogram of %s: $\mu=%d$, $\sigma=%d$' % (col, mu, sigma))



# # Tweak spacing to prevent clipping of ylabel

# fig.tight_layout()

# plt.show()
# fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(20, 20))



# count = 1

# for i in range(4):

#     for j in range(10):

#         show_hist(copy, 'word-' + str(count),fig, axes[j][i])

#         count+=1

# plt.show()
def binarization(col):

    for i in range(len(col)):

        cell = col[i]

        if cell >= 1:

            col[i] = 1

        else:

            col[i] = 0

    return col
# X = X.apply(binarization, axis=1)
sns.countplot(x='Result',data=pd.DataFrame({'Result': y}))
from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import roc_curve

from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.combine import SMOTETomek

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB
X_all = X.merge(result_zee, on='id')

X = X_all
X = X.query('zee == "train"').drop(['Result', 'zee'] + features, axis=1).values

y = result_zee['Result'].values
kf = KFold(shuffle = True,  n_splits=5)

acc = 0

bacc = 0

fc = 0

for train_index, test_index in kf.split(X):

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    

    ros = RandomOverSampler(random_state=42, sampling_strategy='minority')

    X_res, y_res = ros.fit_resample(X_train, y_train)



    params = {"n_estimators": 300, 'learning_rate': 0.1, 'colsample_bytree': 0.3, 'reg_lambda': 1.5, 'reg_alpha':0.5}

    xgb = XGBClassifier(**params)

#     xgb = GradientBoostingClassifier(loss = 'deviance',learning_rate = 0.01,n_estimators = 1000,max_depth = 5,random_state=55)

#     xgb = LinearSVC()

#     xgb = SGDClassifier()

#     xgb = RandomForestClassifier(n_estimators=100,criterion='gini')

#     xgb = MultinomialNB(alpha=1.9) 

    

    xgb.fit(X_train, y_train)



    y_pred = xgb.predict(X_valid)

    

    current_acc = accuracy_score(y_valid, y_pred)

    bcurrent_acc = balanced_accuracy_score(y_valid, y_pred)

    fcurrect_acc = f1_score(y_valid, y_pred)

    print('ACC:', current_acc, 'BACC:', bcurrent_acc, 'F1:', fcurrect_acc)

    print(confusion_matrix(y_valid, y_pred))

    acc += current_acc

    bacc += bcurrent_acc

    fc += fcurrect_acc

    

print("Accuracy CV ", acc/5)

print("Balance Accuracy CV ", bacc/5)

print("F1 CV ", fc/5)
Z.shape
X_all
Z = X_all.query('zee == "test"').drop(['Result', 'zee'] + features, axis=1).values

prediction = xgb.predict(Z)
len(prediction)
sample_df = pd.read_csv("../input/datmin-joints-2020/sample_submission.csv")

sample_df['Result'] = prediction.astype(int)

sample_df.head(9)
sample_df.to_csv("prediction.csv", index=False)