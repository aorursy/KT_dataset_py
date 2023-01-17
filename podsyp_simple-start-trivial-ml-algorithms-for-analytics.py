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
import warnings

warnings.filterwarnings('ignore')

import datetime



import pandas_summary as ps

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif, chi2

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('use_inf_as_na', True)

random_state = 17
full_df = pd.read_csv('/kaggle/input/how-to-do-product-analytics/product.csv', sep=',')

full_df.shape
full_df.head()
full_df['time'] = full_df['time'].apply(pd.to_datetime)
dfs = ps.DataFrameSummary(full_df)

print('categoricals: ', dfs.categoricals.tolist())

print('numerics: ', dfs.numerics.tolist())

dfs.summary()
full_df.info()
full_df.isnull().sum()
for i in ('order_id', 'page_id'):

    full_df[i] = full_df[i].fillna(0).apply(lambda x: x if x == 0 else 1)

    print(full_df.groupby([i]).size().sort_values(ascending=False).head(2))

    print('\n')

print(full_df.groupby(['title']).size().sort_values(ascending=False).head())

sns.countplot('title', data=full_df)

plt.show();

full_df.drop(['order_id', 'page_id'], axis=1, inplace=True)
full_df.head()
full_df = full_df.assign(num_conversion=full_df.groupby(['user_id'])['time'].rank(method='first', ascending=True))

sns.countplot('num_conversion', data=full_df)

plt.show();

full_df['IsBanner_click'] = full_df['title'].apply(lambda x: 1 if x == 'banner_click' else 0)

full_df['IsBanner_click'] = full_df.groupby('user_id').IsBanner_click.transform(np.mean).apply(lambda x: 0 if x == 0 else 1)

full_df['IsFirst_conversion'] = full_df['num_conversion'].apply(lambda x: 1 if x == 1 else 0)

full_df.drop(['user_id', 'title'], axis=1, inplace=True)
full_df.head()
full_df['time_IsMorning'] = full_df['time'].apply(lambda ts: 1 if (ts.hour >= 6) and (ts.hour < 10) else 0)

full_df['time_IsDaylight'] = full_df['time'].apply(lambda ts: 1 if (ts.hour >= 10) and (ts.hour < 16) else 0)

full_df['time_IsEvening'] = full_df['time'].apply(lambda ts: 1 if (ts.hour >= 16) and (ts.hour < 23) else 0)

full_df['time_Hour'] = full_df['time'].apply(lambda ts: ts.hour)

full_df['time_Day'] = full_df['time'].apply(lambda ts: ts.day)

full_df['time_Week_Day'] = full_df['time'].apply(lambda ts: datetime.date(ts.year, ts.month, ts.day).weekday() + 1)

full_df['time_Year_Month'] = full_df['time'].apply(lambda ts: ts.year * 100 + ts.month)

full_df.drop(['time'], axis=1, inplace=True)
full_df.head()
for i in ['product', 'site_version']:

    print('\n')

    print(full_df.groupby([i]).size().sort_values(ascending=False).head())

    sns.countplot(i, data=full_df)

    plt.show()
full_df['SV_IsMobile'] = full_df['site_version'].map({'desktop': 0, 'mobile': 1})

full_df.drop(['site_version'], axis=1, inplace=True)
full_df = pd.get_dummies(full_df, columns=['product'])
full_df.head()
dfs = ps.DataFrameSummary(full_df)

print('categoricals: ', dfs.categoricals.tolist())

print('numerics: ', dfs.numerics.tolist())

dfs.summary()
sns.countplot('target', data=full_df)

plt.show()
access_df = full_df.drop(['product_clothes', 'product_company', 'product_sneakers', 'product_sports_nutrition'], axis=1)
# Number of data points in the minority class

number_records_fraud = len(full_df[full_df.target == 1])

fraud_indices = np.array(full_df[full_df.target == 1].index)



# Picking the indices of the normal classes

normal_indices = full_df[full_df.target == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)

random_normal_indices = np.array(random_normal_indices)



# Appending the 2 indices

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])



# Under sample dataset

under_sample_data = full_df.iloc[under_sample_indices,:]



X = under_sample_data.ix[:, under_sample_data.columns != 'target']

y = under_sample_data.ix[:, under_sample_data.columns == 'target']



# Showing ratio

print("Perc. of banner click or show result: ", len(under_sample_data[under_sample_data.target == 0])/len(under_sample_data))

print("Perc. of order result: ", len(under_sample_data[under_sample_data.target == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))

sns.countplot('target', data=y)

plt.show()
X.shape, y.shape
X.head()
X.describe()
stand_X = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
class Feat_Importance:

    df = None

    columns = None

    random_state = None

    ranks = {}

        

    def __init__(self, X, y, columns, random_state=56, show_dict='N', show_plot='N'):

        self.X = X

        self.y = y

        self.names = columns

        self.random_state = random_state

        self.show_dict = show_dict

        self.show_plot = show_plot

        

    def __rank_to_dict(self, ranks, names, order=1):

        minmax = MinMaxScaler()

        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]

        ranks = map(lambda x: round(x, 2), ranks)

        return dict(zip(names, ranks))

    

    def feat_stats(self):

        self.ranks = {}

        self.get_KBest()

        self.get_LogReg()

        self.get_XGBC()

        

    def get_KBest(self):

        selector = SelectKBest(f_classif)

        selector.fit(self.X, self.y)

        scores = selector.scores_

        scores = pd.Series(scores).fillna(0)

        self.ranks["KBest"] = self.__rank_to_dict(scores, self.names)

        if self.show_dict == 'Y': 

            print('===== KBest dict =====\n', self.ranks["KBest"], '\n\n\n')

        if self.show_plot == 'Y': 

            print('===== KBest plot =====\n', self.X.shape)

            plt.bar(range(len(self.names)), -np.log10(selector.pvalues_))

            plt.xticks(range(len(self.names)), self.names, rotation='vertical');

            

    def get_LogReg(self):

        model_LogRegRidge = LogisticRegression(penalty='l2', C=0.15, 

                                               random_state=self.random_state, solver='liblinear', 

                                               n_jobs=-1)

        model_LogRegRidge.fit(self.X, self.y)

        self.ranks["LogRegRidge"] = self.__rank_to_dict(list(map(float, 

                                    model_LogRegRidge.coef_.reshape(len(self.names), -1))),

                                    self.names, order=1)

        

        if self.show_dict == 'Y': 

            print('===== LogRegRidge dict =====\n', self.ranks["LogRegRidge"], '\n\n\n')

        if self.show_plot == 'Y':

            print('===== LogRegRidge plot =====\n', self.X.shape)

            listsRidge = sorted(self.ranks["LogRegRidge"].items(), key=operator.itemgetter(1))

            dfRidge = pd.DataFrame(np.array(listsRidge).reshape(len(listsRidge), 2),

                       columns=['Features', 'Ranks']).sort_values('Ranks')

            dfRidge['Ranks'] = dfRidge['Ranks'].astype(float)

            dfRidge.plot.bar(x='Features', y='Ranks', color='blue')

            plt.xticks(rotation=90)

    

    def get_XGBC(self):

        model_XGBC = XGBClassifier(objective='binary:logistic',

                           max_depth=7, min_child_weight=5,

                           gamma=0, random_state=random_state, n_jobs=-1,

                           learning_rate=0.1, n_estimators=200)

        model_XGBC.fit(self.X, self.y)

        self.ranks["XGBC"] = self.__rank_to_dict(model_XGBC.feature_importances_, self.names)

        if self.show_dict == 'Y': 

            print('===== XGBClassifier dict =====\n', self.ranks["XGBC"], '\n\n\n')

        if self.show_plot == 'Y':

            print('===== XGBClassifier plot =====\n', self.X.shape)

            plot_importance(model_XGBC)

            plt.show()

    

    def stats_df(self):

        r = {}

        for name in self.names:

            r[name] = round(np.mean([self.ranks[method][name] for method in self.ranks.keys()]), 2)

        methods = sorted(self.ranks.keys())

        self.ranks['Mean'] = r

        methods.append('Mean')



        row_index, AllFeatures_columns = 0, ['Feature', 'Scores']

        AllFeats = pd.DataFrame(columns=AllFeatures_columns)

        for name in self.names:

            AllFeats.loc[row_index, 'Feature'] = name

            AllFeats.loc[row_index, 'Scores'] = [self.ranks[method][name] for method in methods]

            row_index += 1

        AllFeatures_only = pd.DataFrame(AllFeats.Scores.tolist(), )

        AllFeatures_only.rename(columns={0: 'KBest', 1: 'LogRegRidge', 2: 'XGB Classifier', 3: 'Mean'}, inplace=True)

        AllFeatures_only = AllFeatures_only[['KBest', 'LogRegRidge', 'XGB Classifier', 'Mean']]

        AllFeatures_compare = AllFeats.join(AllFeatures_only).drop(['Scores'], axis=1)

        return AllFeatures_compare

    

    def simple_test(self):

        x_train, x_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=0.3, random_state=random_state+37)

        mods = ('BernoulliNB', 'KNeighborsClassifier', 'RandomForestClassifier')

        for nu, model in enumerate([BernoulliNB(), KNeighborsClassifier(n_jobs=-1), 

                      RandomForestClassifier(n_jobs=-1)]):

            model.fit(x_train, y_train)

            predicted = model.predict(x_valid)

            print(mods[nu])

            print('------ accuracy ------\n', metrics.accuracy_score(y_valid, predicted))

            #print('------ confusion_matrix ------\n', metrics.confusion_matrix(y_valid, predicted))

            print('------ roc_auc_score ------\n', metrics.roc_auc_score(y_valid, predicted))

            print('\n')
fi = Feat_Importance(stand_X, y, stand_X.columns)

fi.feat_stats()

fi_df = fi.stats_df()

display(fi_df.sort_values(by=['Mean'], ascending=[False]))
stand_X = stand_X[fi_df[fi_df['Mean'] > 0.21].Feature.values]

stand_X.shape
fi = Feat_Importance(stand_X, y, stand_X.columns)

fi.feat_stats()

fi_df = fi.stats_df()

display(fi_df.sort_values(by=['Mean'], ascending=[False]))
fi.simple_test()
plt.figure(figsize=(10, 8))

sns.heatmap(stand_X.corr(), xticklabels=stand_X.columns, yticklabels=stand_X.columns)
stand_X = stand_X.drop(['IsFirst_conversion'], axis=1)
plt.figure(figsize=(8, 8))

sns.heatmap(stand_X.corr(), xticklabels=stand_X.columns, yticklabels=stand_X.columns)
fi = Feat_Importance(stand_X, y, stand_X.columns)

fi.feat_stats()

fi_df = fi.stats_df()

display(fi_df.sort_values(by=['Mean'], ascending=[False]))
fi.simple_test()
x_train, x_valid, y_train, y_valid = train_test_split(stand_X, y, test_size=0.3, random_state=random_state)
r_for = RandomForestClassifier()

print(r_for)

for_params = {'max_depth': np.arange(4, 10), 'max_features': np.arange(0.25, 0.5, 1), 'n_estimators': [30, 50, 60]}

for_grid = GridSearchCV(r_for, for_params, cv=2, n_jobs=-1)

for_grid.fit(x_train, y_train)

print('best score / best params: ', for_grid.best_score_, for_grid.best_params_)

y_pred = for_grid.predict(x_valid)

print('classification_report: \n', metrics.classification_report(y_pred, y_valid))

print('accuracy_score: ', metrics.accuracy_score(y_pred, y_valid))

print('roc_auc_score: ', metrics.roc_auc_score(y_pred, y_valid))
log_r = LogisticRegression()

grid_values = {'penalty': ['l2'], 'C': [0.0001, 0.001, 0.01, 0.1]}

lr_grid = GridSearchCV(log_r, param_grid=grid_values, cv=2, n_jobs=-1)

lr_grid.fit(x_train, y_train)

print('best score / best params: ', lr_grid.best_score_, lr_grid.best_params_)

y_pred = lr_grid.predict(x_valid)

print('classification_report: \n', metrics.classification_report(y_pred, y_valid))

print('accuracy_score: ', metrics.accuracy_score(y_pred, y_valid))

print('roc_auc_score: ', metrics.roc_auc_score(y_pred, y_valid))
xgb_m = XGBClassifier()

xgb_params = [

    {"n_estimators": [300, 350],

     "max_depth": [3,  5],

     "learning_rate": [0.01, 0.05]}

]

xgb_grid = GridSearchCV(xgb_m, xgb_params, cv=2, refit=True, verbose=1, n_jobs=-1)

xgb_grid.fit(x_train, y_train)

print('best score / best params: ', xgb_grid.best_score_, xgb_grid.best_params_)

y_pred = xgb_grid.predict(x_valid)

print('classification_report: \n', metrics.classification_report(y_pred, y_valid))

print('accuracy_score: ', metrics.accuracy_score(y_pred, y_valid))

print('roc_auc_score: ', metrics.roc_auc_score(y_pred, y_valid))
cluster_stand_df = stand_X

cluster_stand_df['target'] = preprocessing.scale(y)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=random_state).fit(cluster_stand_df)

unique, counts = np.unique(kmeans.labels_, return_counts=True)

print(np.asarray((unique, counts)).T)
cluster_df = X[[i for i in stand_X.columns if i != 'target']]

cluster_df['target'] = y

cluster_df['Cluster'] = kmeans.labels_ # 497444
cluster_df['Cluster'].value_counts(normalize=True)
cluster_df.groupby('Cluster').mean().sort_values(by=['target'], ascending=False)
ax = sns.violinplot(x='target', y='Cluster',

                         data=cluster_df, height=4, aspect=.7)

plt.show();
display(fi_df.sort_values(by=['Mean'], ascending=[False]))
ax = sns.violinplot(x='target', y='num_conversion',

                         data=cluster_df[cluster_df['num_conversion'] <= 10], height=4, aspect=.7)

plt.show();
feat = [f for f in cluster_df.columns if 'Is' in f]

for i in feat:

    print(i)

    plt.figure()

    tmp = cluster_df[cluster_df[i] == 1]

    tmp['target'].hist(figsize=(6, 3), bins=2, color = 'red')

    plt.show();
feat = [f for f in cluster_df.columns if 'time' in f]

for i in feat:

    print(i)

    plt.figure()

    cluster_df[cluster_df['target'] == 1][i].hist(figsize=(5, 3), alpha=0.4, color = 'green')

    cluster_df[cluster_df['target'] == 0][i].hist(figsize=(5, 3), alpha=0.4, color = 'red')

    plt.show();
feat = [f for f in cluster_df.columns if 'product' in f and '_company' not in f]

for i in feat:

    print(i)

    plt.figure()

    tmp = cluster_df[cluster_df[i] == 1]

    tmp['target'].hist(figsize=(6, 3), bins=3)

    plt.show();