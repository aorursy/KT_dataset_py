import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
# setting up default plotting parameters

%matplotlib inline



plt.rcParams['figure.figsize'] = [20.0, 7.0]

plt.rcParams.update({'font.size': 22,})



sns.set_palette('viridis')

sns.set_style('white')

sns.set_context('talk', font_scale=0.8)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print('Train Shape: ', train.shape)

print('Test Shape: ', test.shape)



train.head()
# using seaborns countplot to show distribution of questions in dataset

fig, ax = plt.subplots()

g = sns.countplot(train.target, palette='viridis')

g.set_xticklabels(['0', '1'])

g.set_yticklabels([])



# function to show values on bars

def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

show_values_on_bars(ax)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Distribution of Target', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# prepare for modeling

X_train_df = train.drop(['id', 'target'], axis=1)

y_train = train['target']



X_test = test.drop(['id'], axis=1)



# scaling data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_df)

X_test = scaler.transform(X_test)
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(n_estimators=100)



lr_scores = cross_val_score(lr,

                            X_train,

                            y_train,

                            cv=5,

                            scoring='roc_auc')

rfc_scores = cross_val_score(rfc, X_train, y_train, cv=5, scoring='roc_auc')



print('LR Scores: ', lr_scores)

print('RFC Scores: ', rfc_scores)
# checking which are the most important features

feature_importance = rfc.fit(X_train, y_train).feature_importances_



# Make importances relative to max importance.

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[-20:-1:1]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train_df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Feature Importance', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

sns.despine(left=True, bottom=True)

plt.show()
# check missing values

train.isnull().any().any()
from sklearn import feature_selection



sel = feature_selection.VarianceThreshold()

train_variance = sel.fit_transform(train)

train_variance.shape
# find correlations to target

corr_matrix = train.corr().abs()



print(corr_matrix['target'].sort_values(ascending=False).head(10))
# Select upper triangle of correlation matrix

matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

sns.heatmap(matrix)

plt.show;
# Find index of feature columns with high correlation

to_drop = [column for column in matrix.columns if any(matrix[column] > 0.50)]

print('Columns to drop: ' , (len(to_drop)))
# feature extraction

k_best = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=100)

# fit on train set

fit = k_best.fit(X_train, y_train)

# transform train set

univariate_features = fit.transform(X_train)
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(n_estimators=100)



lr_scores = cross_val_score(lr, univariate_features, y_train, cv=5, scoring='roc_auc')

rfc_scores = cross_val_score(rfc, univariate_features, y_train, cv=5, scoring='roc_auc')



print('LR Scores: ', lr_scores)

print('RFC Scores: ', rfc_scores)
# checking which are the most important features

feature_importance = rfc.fit(univariate_features, y_train).feature_importances_



# Make importances relative to max importance.

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[-20:-1:1]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train_df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Feature Importance', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

sns.despine(left=True, bottom=True)

plt.show()
# feature extraction

rfe = feature_selection.RFE(lr, n_features_to_select=100)



# fit on train set

fit = rfe.fit(X_train, y_train)



# transform train set

recursive_features = fit.transform(X_train)
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(n_estimators=10)



lr_scores = cross_val_score(lr, recursive_features, y_train, cv=5, scoring='roc_auc')

rfc_scores = cross_val_score(rfc, recursive_features, y_train, cv=5, scoring='roc_auc')



print('LR Scores: ', lr_scores)

print('RFC Scores: ', rfc_scores)
# checking which are the most important features

feature_importance = rfc.fit(recursive_features, y_train).feature_importances_



# Make importances relative to max importance.

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[-20:-1:1]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train_df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Feature Importance', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

sns.despine(left=True, bottom=True)

plt.show()
# feature extraction

select_model = feature_selection.SelectFromModel(lr)



# fit on train set

fit = select_model.fit(X_train, y_train)



# transform train set

model_features = fit.transform(X_train)
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(n_estimators=100)



lr_scores = cross_val_score(lr, model_features, y_train, cv=5, scoring='roc_auc')

rfc_scores = cross_val_score(rfc, model_features, y_train, cv=5, scoring='roc_auc')



print('LR Scores: ', lr_scores)

print('RFC Scores: ', rfc_scores)
# checking which are the most important features

feature_importance = rfc.fit(model_features, y_train).feature_importances_



# Make importances relative to max importance.

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[-20:-1:1]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train_df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Feature Importance', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

sns.despine(left=True, bottom=True)

plt.show()
from sklearn.decomposition import PCA

# pca - keep 90% of variance

pca = PCA(0.90)



principal_components = pca.fit_transform(X_train)

principal_df = pd.DataFrame(data = principal_components)

principal_df.shape
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(n_estimators=100)



lr_scores = cross_val_score(lr, principal_df, y_train, cv=5, scoring='roc_auc')

rfc_scores = cross_val_score(rfc, principal_df, y_train, cv=5, scoring='roc_auc')



print('LR Scores: ', lr_scores)

print('RFC Scores: ', rfc_scores)
# pca keep 75% of variance

pca = PCA(0.75)

principal_components = pca.fit_transform(X_train)

principal_df = pd.DataFrame(data = principal_components)

principal_df.shape
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(n_estimators=100)



lr_scores = cross_val_score(lr, principal_df, y_train, cv=5, scoring='roc_auc')

rfc_scores = cross_val_score(rfc, principal_df, y_train, cv=5, scoring='roc_auc')



print('LR Scores: ', lr_scores)

print('RFC Scores: ', rfc_scores)
# checking which are the most important features

feature_importance = rfc.fit(principal_df, y_train).feature_importances_



# Make importances relative to max importance.

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[-20:-1:1]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train_df.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Feature Importance', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

sns.despine(left=True, bottom=True)

plt.show()
# feature extraction

rfe = feature_selection.RFE(lr, n_features_to_select=100)



# fit on train set

fit = rfe.fit(X_train, y_train)



# transform train set

recursive_X_train = fit.transform(X_train)

recursive_X_test = fit.transform(X_test)



lr = LogisticRegression(C=1, class_weight={1:0.6, 0:0.4}, penalty='l1', solver='liblinear')

lr_scores = cross_val_score(lr, recursive_X_train, y_train, cv=5, scoring='roc_auc')

lr_scores.mean()
predictions = lr.fit(recursive_X_train, y_train).predict_proba(recursive_X_test)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = predictions

submission.to_csv('submission.csv', index=False)

submission.head()