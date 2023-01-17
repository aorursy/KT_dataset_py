# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.info()
columns_to_drop = ['AGMTNO', 'AGMT_DATE', 'DEALER_TYPE', 'EMPLOYMENT_TYPE', 'RESIDENT_TYPE', 
                   'ADDRESS_TYPE', 'FLOOR_TYPE', 'ADVANCE_EMI_COUNT', 'IF_PRESENT_PASSPORT',
                  'MORTAGE']
x_train = train_df.drop(columns_to_drop, axis = 1, inplace = False)
x_train.IS_EXISTING_CUSTOMER.fillna('N', inplace=True)
x_train.dropna(inplace = True)
x_train = x_train.apply(lambda col: pd.factorize(col, sort=True)[0])
import seaborn as sns
import matplotlib.pyplot as plt

corr = x_train.corr()

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
y_train = x_train.iloc[:,-1]
x_train.drop('DEFAULTED (dependent variable)',axis = 1, inplace = True)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.25,
                                                     random_state = 42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state = 42)
model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_valid, model.predict(x_valid)))
feature_importance = pd.Series(model.feature_importances_, 
                               index=x_train.columns)
feature_importance.sort_values(inplace=True, ascending=True)
feature_importance.plot(kind='barh', figsize=(10,13))
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

normalize = StandardScaler().fit_transform(x_train)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_train)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal-component 1', 'principal-component 2'])

principalDf.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
pca_df = pd.concat([principalDf, y_train], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['b', 'y']
for target, color in zip(targets, colors):
    indicesToKeep = pca_df['DEFAULTED (dependent variable)'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'principal-component 1']
               , pca_df.loc[indicesToKeep, 'principal-component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
results = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]
for trees in n_estimator_options:
    model = RandomForestClassifier(trees, n_jobs=-1,
                                   random_state=42)
    model.fit(x_train, y_train)
    results.append(accuracy_score(y_valid, model.predict(x_valid)))
pd.Series(results, n_estimator_options).plot()
results = []
max_features_options = [None, "sqrt", "log2", 0.9, 0.2]
for max_features in max_features_options:
    model = RandomForestClassifier(n_estimators=1000, n_jobs=-1,
                                   random_state=42, max_features=max_features)
    model.fit(x_train, y_train)
    results.append(accuracy_score(y_valid, model.predict(x_valid)))
pd.Series(results, max_features_options).plot(kind='barh')
results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for min_samples in min_samples_leaf_options:
    model = RandomForestClassifier(n_estimators=1000, n_jobs=-1,
                                   random_state=42, max_features=None,
                                  min_samples_leaf=min_samples)
    model.fit(x_train, y_train)
    results.append(accuracy_score(y_valid, model.predict(x_valid)))
pd.Series(results, min_samples_leaf_options).plot()
model = RandomForestClassifier(n_estimators=500,
                              n_jobs=-1,
                              random_state=42,
                              max_features=None,
                              min_samples_leaf=4)
model.fit(x_train, y_train)
print(accuracy_score(y_valid, model.predict(x_valid)))
test_df = pd.read_csv('../input/Validation_v2.csv')
test_index = test_df.AGMTNO
test_df.drop(columns_to_drop, axis = 1, inplace = True)
x_test = test_df.apply(lambda col: pd.factorize(col, sort=True)[0])
y_test = model.predict_proba(x_test)
final_predict = []
for value in y_test:
    final_predict.append(value[1])
from sklearn.metrics import roc_curve, roc_auc_score

print (roc_auc_score(y_valid, model.predict_proba(x_valid)[:,1]))

fpr, tpr, _ = roc_curve(y_valid, model.predict(x_valid))

plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
final_df = pd.concat([pd.DataFrame(test_index, columns=['AGMTNO']), 
                      pd.DataFrame(final_predict, columns=['Prob to default'])], 
                      axis=1)
final_df.to_csv('mysubmission.csv', index = False)
final_df
