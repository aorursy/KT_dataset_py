import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
bc = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
bc.info()
bc.describe()
bc.head()
bc.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
bc.isnull().sum().sum()
bc.isnull().any(axis=1).sum()
bc['diagnosis'].value_counts()
sns.countplot(bc['diagnosis'])
bc.groupby('diagnosis').mean()
plt.figure(figsize=(15,5))

plt.plot((bc.groupby('diagnosis').mean().loc['M'])/((bc.groupby('diagnosis').mean().loc['B'])))

plt.title('Ratio of Malignant to Benign values')

plt.xticks(rotation=90)

plt.show()
X = bc.iloc[:, 1:]

y = bc.iloc[:, :1]
X1 = X.iloc[:, :15]
X1 = (X1-X1.mean())/X1.std()
data = pd.concat([y, X1], axis=1)
data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
plt.figure(figsize=(20,7))

sns.swarmplot(x='features', y='value', hue='diagnosis', data=data)

plt.xticks(rotation=45)

plt.show()
X2 = X.iloc[:, 15:]
X2 = (X2-X2.mean())/X2.std()
data = pd.concat([y, X2], axis=1)
data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
plt.figure(figsize=(20,7))

sns.swarmplot(x='features', y='value', hue='diagnosis', data=data)

plt.xticks(rotation=45)

plt.show()
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
def processing(df):

    X = df.iloc[ :, 1:]

    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale = MinMaxScaler()

    X_train = scale.fit_transform(X_train)

    X_test = scale.transform(X_test)

    X_train = pd.DataFrame(X_train)

    X_test = pd.DataFrame(X_test)

    return X_train, X_test, y_train, y_test 
def random_forest(df, i):

    X = df.iloc[ :, 1:]

    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    scale = MinMaxScaler()

    X_train = scale.fit_transform(X_train)

    X_test = scale.transform(X_test)

    rfc = RandomForestClassifier(n_estimators=100)

    rfc.fit(X_train, y_train)

    return rfc, X_test, y_test
def scores(f, df, n):

    """

        f is machine learning algorithm funtion

        df is dataframe to given to ml funtion

        n is number of random states used for splitting the dataframe

        this function returns array of scores for random states 0 to n.

    """

    scores = []

    for i in range(n):

        clf, X_test, y_test = f(df, i)

        scores.append(clf.score(X_test, y_test))

    return np.array(scores)
score_list = scores(random_forest, bc, 100)
plt.figure(figsize=(15,5))

plt.plot(score_list)

plt.xlabel('random state')

plt.ylabel('mean accuracy score')
score_list.mean()
rfc, X_test, y_test = random_forest(bc, 109)
predictions = rfc.predict(X_test)
accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
fig, axes = plt.subplots(1,1, figsize=(15,5))

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, ax=axes)
pred_prob = rfc.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:, 1], pos_label='M')
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:, 1], pos_label='M')
roc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label='(area = %0.2f)' % roc_score)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5), gridspec_kw={'width_ratios': [1, 2]})

sns.heatmap(pred_prob[np.argsort(pred_prob[:, 0])], ax=ax1)

ax2.plot(pred_prob[np.argsort(pred_prob[:, 0])])

ax2.set_xlabel('test case number')

ax2.set_ylabel('probability')

ax2.legend(['B', 'M'])
pred_prob[(pred_prob[:, 0] < .8) & (pred_prob[:, 1] < .8)]
percentage = pred_prob[(pred_prob[:, 0] < .8) & (pred_prob[:, 1] < .8)].shape[0]/pred_prob.shape[0]
f'{np.round(percentage*100, 2)} percentage of the predictions have less than 80 % of accuracy'
plt.figure(figsize=(15,10))

sns.heatmap(bc.corr(), cmap='Spectral', annot=True, fmt='.1f')
lst = ['diagnosis', 'texture_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean','texture_se',

       'area_se', 'smoothness_se','compactness_se', 'symmetry_se','fractal_dimension_se', 'texture_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'symmetry_worst', 'fractal_dimension_worst']
bc_1 = bc.loc[:, lst]
plt.figure(figsize=(15,10))

sns.heatmap(bc_1.corr(), cmap='Spectral', annot=True)
data = bc_1.iloc[:, 1:]
data = (data-data.mean())/data.std()
data = pd.concat([bc['diagnosis'], data], axis=1)
data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
plt.figure(figsize=(20,7))

sns.violinplot(x='features', y='value', hue='diagnosis', data=data, split=True, inner='quart')

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,7))

sns.swarmplot(x='features', y='value', hue='diagnosis', data=data)

plt.xticks(rotation=45)

plt.show()
score_list_1 = scores(random_forest, bc_1, 100)
score_1 = np.round(score_list_1.mean(), 2)
score_1
rfc, X_test, y_test = random_forest(bc_1, 34)
y = bc['diagnosis']
rfc.score(X_test, y_test)
sns.heatmap(confusion_matrix(y_test, rfc.predict(X_test)), annot=True)
feature_imp = pd.DataFrame(rfc.feature_importances_, bc_1.iloc[:, 1:].columns, columns=['importance']).sort_values(by='importance', ascending=False)
feature_imp.head()
rfc_df = pd.concat([bc['diagnosis'], bc_1[feature_imp.head().index]], axis=1)
score_list_fs_1 = scores(random_forest, rfc_df , 100)
score_fs_1 = np.round(score_list_fs_1.mean(), 2)
score_fs_1
F"With five important features we can get the score of {score_fs_1} compared to {score_1} with 18 features."
rfc, X_test, y_test = random_forest(rfc_df, 34)
sns.heatmap(confusion_matrix(y_test, rfc.predict(X_test)), annot=True)
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV
X_train, X_test, y_train, y_test = train_test_split(bc.iloc[:, 1:], bc['diagnosis'], test_size=0.2, random_state=23)
min_ = X_train.min()

max_ = X_train.max()
X_train = (X_train - min_)/(max_ - min_)
X_test = (X_test - min_)/(max_ - min_)
best_features = SelectKBest(chi2, k=5).fit(X_train, y_train)
X_train.columns[best_features.get_support(indices=True)]
score_list = scores(random_forest, pd.concat([bc['diagnosis'], bc[X_train.columns[best_features.get_support(indices=True)]]], axis=1), 100)
score_list.mean()
f'Score with features selected by chi2 {np.round(score_list.mean(), 2)}'
rfecv = RFECV(RandomForestClassifier(), min_features_to_select=5)
rfecv.fit(X_train, y_train)
X_train.columns[rfecv.support_]
rfecv.n_features_
from sklearn.decomposition import PCA
X_train, X_test, y_train, y_test = train_test_split(bc.iloc[:, 1:], bc['diagnosis'], test_size=0.2, random_state=101)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
pca = PCA()
pca.fit(X_train)
pca.explained_variance_ratio_
plt.figure(figsize=(15,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xticks(np.arange(0, 31))

plt.grid()

plt.xlabel('n_components')

plt.ylabel('cumulative explained_variance_ratio_')