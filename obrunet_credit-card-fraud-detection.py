import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



pd.set_option('display.max_columns', 100)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, classification_report, auc, precision_recall_curve

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC
import lightgbm as lgbm

import xgboost as xgb
df = pd.read_csv('../input/creditcard.csv')

df.head()
df.shape
df.info()
#df.isnull().sum()
df.duplicated().sum()
#df = df.drop_duplicates()

#df.shape
non_PCA_features = ['Time', 'Amount', 'Class']

df[non_PCA_features].describe()
sns.countplot(x='Class', data=df)

plt.show()
ratio = df.Class.sum() / df.shape[0]

print(f'Percentage of class #1 : {ratio * 100 :.2f}% - Number of frauds is highly imbalanced !')
plt.figure(figsize=(14, 5))

sns.distplot(df['Amount'], label = 'Amount')

plt.show()
print(f"Percentage of transactions' amount below 200€ : {len(df[df.Amount < 200]) / len(df) * 100 :.2f}%")
print(f"Percentage of transactions' amount below 500€ : {len(df[df.Amount < 500]) / len(df) * 100 :.2f}%")
plt.figure(figsize=(14, 8))



plt.subplot(2, 1, 1)

sns.kdeplot(df.loc[df['Class'] == 1, 'Amount'], shade=True, label = 'Fraud')

plt.subplot(2, 1, 2)

sns.kdeplot(df.loc[df['Class'] == 0, 'Amount'], shade=True, label = 'Normal')

plt.show()
plt.figure(figsize=(14, 5))



sns.kdeplot(df.loc[df['Class'] == 1, 'Amount'], shade=True, label = 'Fraud')

sns.kdeplot(df.loc[df['Class'] == 0, 'Amount'], shade=True, label = 'Normal')

plt.show()
print(f'nb of days during when the data were collected : {df.Time.max() / (60 * 60 * 24):.2f}')
df['Hours'] = round((df['Time'] / (60 * 60)) % 24)

df.Hours = df.Hours.astype(int)
df['Days'] = round(df['Time'] / (60 * 60 * 24))

df.Days = df.Days.astype(int)
df.Days.value_counts()
df.Class.sum()
plt.figure(figsize=(14, 6))

sns.distplot((df['Time'] / (60 * 60)), label = 'All transactions', bins=24)

plt.title('Transaction distribution during 2 days')

plt.show()
plt.figure(figsize=(14, 6))

sns.kdeplot((df['Time'] / (60 * 60)) % 24, label = 'All transactions', shade=True)

sns.kdeplot((df.loc[df['Class'] == 1, 'Time'] / (60 * 60)) % 24, label = 'Fraud', shade=True)

plt.title('Transaction Nb per hour during a typical day')

plt.show()
plt.figure(figsize=(14, 6))

sns.kdeplot((df['Time'] / (60 * 60)), label = 'All transactions', shade=True)

sns.kdeplot((df.loc[df['Class'] == 1, 'Time'] / (60 * 60)), label = 'Fraud', shade=True)

plt.title('Transaction Nb per hour during 2 days')

plt.show()
corr = df.corr()

#corr
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
pca_feat = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',

       'Hours']

y = df.Class

X_centered = df[pca_feat]
scaler = StandardScaler()

X_centered = scaler.fit_transform(X_centered)
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_centered)

X_pca[:5]
# Then we plot the results of PCA

plt.figure(figsize=(8, 6))

plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'o', label='Genuine')

plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], '*', label='Fraud')

plt.legend(loc=0)

plt.title('PCA 2D projection')

plt.show()
from sklearn.utils import resample



X_sub, y_sub = resample(X_pca, y, replace=False, n_samples=2000, random_state=0)
tsne = TSNE(n_components=2)



# Here we perform the t-SNE

X_tsne = tsne.fit_transform(X_sub)



# Then we plot the results of t-SNE

plt.plot(X_tsne[y_sub == 0, 0], X_tsne[y_sub == 0, 1], 'o', label='Genuine')

plt.plot(X_tsne[y_sub == 1, 0], X_tsne[y_sub == 1, 1], '*', label='Fraud')

plt.legend(loc=0)

plt.title('t-SNE projection')

plt.show()
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_centered)
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X_pca[y == 0, 2], X_pca[y == 0, 1], X_pca[y == 0, 0], 'o', label='Genuine')

ax.scatter(X_pca[y == 1, 2], X_pca[y == 1, 1], X_pca[y == 1, 0], '*', label='Fraud')



plt.legend()

plt.show()
tsne = TSNE(n_components=3)

X_tsne = tsne.fit_transform(X_sub)



fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X_tsne[y_sub == 0, 0], X_tsne[y_sub == 0, 1], X_tsne[y_sub == 0, 2], 'o', label='Genuine')

ax.scatter(X_tsne[y_sub == 1, 0], X_tsne[y_sub == 1, 1], X_tsne[y_sub == 1, 2], '*', label='Fraud')



plt.legend()

plt.show()
features_kept = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',

       'Hours', 'Days']
y = df['Class']

X = df[features_kept]
rnd_clf = RandomForestClassifier(n_jobs=-1)

rnd_clf.fit(X, y)
feature_importances = pd.DataFrame(rnd_clf.feature_importances_, index = X.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances[:10]
plt.figure(figsize=(14, 8))

sns.barplot(x="importance", y=feature_importances.index, data=feature_importances)

plt.show()
X = pd.get_dummies(data=X, columns=['Hours'], drop_first=True)
X.head()
scaler = StandardScaler()

X[['Amount', 'Days']] = scaler.fit_transform(X[['Amount', 'Days']])
X.drop(columns=['Days'])

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Just a test

model = RandomForestClassifier()

model.fit(X_train, y_train)



# predict probabilities

probs = model.predict_proba(X_test)



# keep probabilities for the positive outcome only

probs = probs[:, 1]



# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate precision-recall AUC

auc_score = auc(recall, precision)

auc_score
# f1_score binary by default

def get_f1_scores(fitted_clf, model_name):

    y_train_pred, y_pred = fitted_clf.predict(X_train), fitted_clf.predict(X_test)

    print(model_name, ' :')

    print(f'Training F1 score = {f1_score(y_train, y_train_pred) * 100:.2f}% / Test F1 score = {f1_score(y_test, y_pred)  * 100:.2f}%')

    print(classification_report(y_test, y_pred))
def get_auc_scores(fitted_clf, model_name):

    print(model_name, ' :')

    

    # get classes predictions for the classification report 

    y_train_pred, y_pred = fitted_clf.predict(X_train), fitted_clf.predict(X_test)

    print(classification_report(y_test, y_pred), '\n') # target_names=y

    

    # computes probabilities keep the ones for the positive outcome only    

    probs = fitted_clf.predict_proba(X_test)[:, 1]

    

    # calculate precision-recall curve

    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    

    # calculate precision-recall AUC

    auc_score = auc(recall, precision)

    print(f'Area Under the Precision-Recall Curve (AUPRC) = {auc_score * 100 :.2f}%')
model = RandomForestClassifier()

model.fit(X_train, y_train)

get_auc_scores(model, 'RandomForest not weighted')
y_train.sum(), len(y_train) - y_train.sum()
model = RandomForestClassifier(class_weight={1:398, 0:227447})

model.fit(X_train, y_train)

get_auc_scores(model, 'RandomForest weighted')
model = lgbm.LGBMClassifier(n_jobs = -1)

model.fit(X_train, y_train)

get_auc_scores(model, 'LGBM non weighted')
model = lgbm.LGBMClassifier(n_jobs = -1, class_weight={0:398, 1:227447})

model.fit(X_train, y_train)

get_auc_scores(model, 'LGBM weighted')
model = xgb.XGBClassifier(objective="binary:logistic")

model.fit(X_train, y_train)

get_auc_scores(model, 'XGB without ratio')
ratio = ((len(y_train) - y_train.sum()) - y_train.sum()) / y_train.sum()

ratio
model = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=ratio)

model.fit(X_train, y_train)

get_auc_scores(model, 'XGB with ratio')
# conda install -c conda-forge imbalanced-learn

from imblearn.over_sampling import SMOTE
X_train.shape, y_train.shape
sm = SMOTE(sampling_strategy='auto', k_neighbors=3, n_jobs=1, ratio=0.01)

X_train, y_train = sm.fit_resample(X_train, y_train)

X_train.shape, y_train.shape
y_train.sum(), len(y_train) - y_train.sum()
model = RandomForestClassifier()

model.fit(X_train, y_train)

get_auc_scores(model, 'RandomForest with SMOTE')
model = RandomForestClassifier(class_weight={1:2274, 0:227437})

model.fit(X_train, y_train)

get_auc_scores(model, 'RandomForest weighted')
model = lgbm.LGBMClassifier(n_jobs = -1)

model.fit(X_train, y_train)

get_auc_scores(model, 'LGBM with SMOTE')
model = lgbm.LGBMClassifier(n_jobs = -1, class_weight={0:2274, 1:227437})

model.fit(X_train, y_train)

get_auc_scores(model, 'LGBM with SMOTE and ratio')
lr = LogisticRegression(C=0.01, penalty='l1').fit(X_train, y_train)

get_auc_scores(lr, 'Logistic Regression')
probs = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, probs)
from inspect import signature

from sklearn.metrics import average_precision_score



average_precision = average_precision_score(y_test, probs)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))



step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})



plt.figure(figsize=(8, 6))

plt.step(recall, precision, color='b', alpha=0.2, where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
data = {'Country': ['Belgium',  'India',  'Brazil'],

    'Capital': ['Brussels',  'New Delhi',  'Brasilia'],

    'Population': [11190846, 1303171035, 207847528]}

test = pd.DataFrame(data,columns=['Country',  'Capital',  'Population'])

test
data = {'Models': ['RandomForrest w/o ratio,  w/o smote',

                   'RandomForrest w/  ratio,  w/o smote',

                   'RandomForrest w/o ratio,  w/  smote',

                   'RandomForrest w/  ratio,  w/  smote',

                   'LGBM          w/o ratio,  w/o smote',

                   'LGBM          w/  ratio,  w/o smote',

                   'LGBM          w/o ratio,  w/  smote',

                   'LGBM          w/  ratio,  w/  smote',

                   'XGBoost       w/o ratio,  w/o smote',

                   'XGBoost       w/  ratio,  w/o smote'],

    'AUPRC': [88.65, 80.41, 88.21, 81.85, 20.58, 79.59, 86.49, 89.91, 86.19, 70.62]}



scores_df = pd.DataFrame(data,columns=['Models',  'AUPRC'])



plt.figure(figsize=(8, 6))

sns.barplot(y="Models", x="AUPRC", data=scores_df)

plt.show()
X_sub, y_sub = resample(X, y, replace=False, n_samples=2000, random_state=0)
X_sub.shape, y_sub.shape
from sklearn.neighbors import LocalOutlierFactor



# fit the model for outlier detection (default)

clf = LocalOutlierFactor(n_neighbors=20, contamination="auto")

# use fit_predict to compute the predicted labels of the training samples

# (when LOF is used for outlier detection, the estimator has no predict,

# decision_function and score_samples methods).



#y_pred = clf.fit_predict(X_train)

#n_errors = (y_pred != y).sum()

#X_scores = clf.negative_outlier_factor_





#plt.figure(figsize=(8,8))

#plt.title("Local Outlier Factor (LOF)")

#plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10., label='Data points')



# plot circles with radius proportional to the outlier scores

#radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

#plt.scatter(X_train[:, 0], X_train[:, 1], s=1000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')

#legend = plt.legend(loc='lower left')

#plt.show()



#print("prediction errors: {}".format(n_errors))

#print("Negative LOF scores: {}".format(clf.negative_outlier_factor_))

#print("Offset (threshold to consider sample as anomaly or not): {}".format(clf.offset_))