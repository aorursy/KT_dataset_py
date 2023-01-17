import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier



import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')



import os

os.listdir('../input')

brainwave_df = pd.read_csv('../input/emotions.csv')
brainwave_df.head()
brainwave_df.tail()
brainwave_df.shape
brainwave_df.dtypes
brainwave_df.describe()
plt.figure(figsize=(12,5))

sns.countplot(x=brainwave_df.label, color='mediumseagreen')

plt.title('Emotional sentiment class distribution', fontsize=16)

plt.ylabel('Class Counts', fontsize=16)

plt.xlabel('Class Label', fontsize=16)

plt.xticks(rotation='vertical');
label_df = brainwave_df['label']

brainwave_df.drop('label', axis = 1, inplace=True)
correlations = brainwave_df.corr(method='pearson')

correlations
skew = brainwave_df.skew()

skew
%%time



pl_random_forest = Pipeline(steps=[('random_forest', RandomForestClassifier())])

scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for RandomForest : ', scores.mean())
%%time



pl_log_reg = Pipeline(steps=[('scaler',StandardScaler()),

                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])

scores = cross_val_score(pl_log_reg, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression: ', scores.mean())
scaler = StandardScaler()

scaled_df = scaler.fit_transform(brainwave_df)

pca = PCA(n_components = 20)

pca_vectors = pca.fit_transform(scaled_df)

for index, var in enumerate(pca.explained_variance_ratio_):

    print("Explained Variance ratio by Principal Component ", (index+1), " : ", var)

plt.figure(figsize=(25,8))

sns.scatterplot(x=pca_vectors[:, 0], y=pca_vectors[:, 1], hue=label_df)

plt.title('Principal Components vs Class distribution', fontsize=16)

plt.ylabel('Principal Component 2', fontsize=16)

plt.xlabel('Principal Component 1', fontsize=16)

plt.xticks(rotation='vertical');
%%time

pl_log_reg_pca = Pipeline(steps=[('scaler',StandardScaler()),

                             ('pca', PCA(n_components = 2)),

                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])

scores = cross_val_score(pl_log_reg_pca, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression with 2 Principal Components: ', scores.mean())
%%time



pl_log_reg_pca_10 = Pipeline(steps=[('scaler',StandardScaler()),

                             ('pca', PCA(n_components = 10)),

                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])

scores = cross_val_score(pl_log_reg_pca_10, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Logistic Regression with 10 Principal Components: ', scores.mean())
%%time



pl_mlp = Pipeline(steps=[('scaler',StandardScaler()),

                             ('mlp_ann', MLPClassifier(hidden_layer_sizes=(1275, 637)))])

scores = cross_val_score(pl_mlp, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for ANN : ', scores.mean())
%%time



pl_svm = Pipeline(steps=[('scaler',StandardScaler()),

                             ('pl_svm', LinearSVC())])

scores = cross_val_score(pl_svm, brainwave_df, label_df, cv=10,scoring='accuracy')

print('Accuracy for Linear SVM : ', scores.mean())
%%time

pl_xgb = Pipeline(steps=

                  [('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])

scores = cross_val_score(pl_xgb, brainwave_df, label_df, cv=10)

print('Accuracy for XGBoost Classifier : ', scores.mean())