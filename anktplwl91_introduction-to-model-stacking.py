# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from catboost import CatBoostClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sc = StandardScaler()
le = LabelEncoder()
onehot = OneHotEncoder(sparse=False)
# Reading train.csv
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.describe()
sns.countplot(train_df['Target'])
plt.show()
# Reading test.csv file
test_df = pd.read_csv('../input/test.csv')
test_df.head()
# Combining both train and test file into one. This will help us preprocessing both files simultaneously, and after we are done with that, we can seperate both.
all_df = train_df.append(test_df, sort=False)
all_df.shape
# Checking fraction of null values in each feature column (ignore the Target variable as its null for the test.csv file)
missing_vals = (all_df.isnull().sum() / len(all_df)).sort_values(ascending=False)
missing_vals = missing_vals[missing_vals > 0]
missing_vals = missing_vals.to_frame()
missing_vals.columns = ['count']
missing_vals.index.names = ['Name']
missing_vals['Name'] = missing_vals.index

sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=missing_vals)
plt.xticks(rotation = 90)
plt.show()
# Dropping columns having too much null values. Filling null values by interpolating in such feature columns can lead to misleading data, so its better to drop them.
# Filling feature columns with too little null values with their median values, as the Target classes are imbalanced, its a good idea to replace null values with
# median values rather than mean values
all_df.drop(['rez_esc', 'v18q1', 'v2a1'], axis=1, inplace=True)
all_df.fillna({'SQBmeaned': all_df['SQBmeaned'].median(), 'meaneduc': all_df['meaneduc'].median()}, inplace=True)
# dividing feature columns according to their dtypes, so that we can visualize them further
float_cols = [col for col in all_df.columns if all_df[col].dtype=='float64']
int_cols = [col for col in all_df.columns if all_df[col].dtype=='int64']
object_cols = [col for col in all_df.columns if all_df[col].dtype=='object']
del(float_cols[-1])
float_flat = pd.melt(all_df, value_vars=float_cols)
g = sns.FacetGrid(float_flat, col='variable', col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
plt.show()
# Log transforming float-type feature columns to remove their skewness using log(1+x)
log_meaneduc = np.log1p(all_df['meaneduc'])
log_overcrowding = np.log1p(all_df['overcrowding'])
log_SQBovercrowding = np.log1p(all_df['SQBovercrowding'])
log_SQBdependency = np.log1p(all_df['SQBdependency'])
log_SQBmeaned = np.log1p(all_df['SQBmeaned'])

temp_df = pd.DataFrame({'log_meaneduc': log_meaneduc, 'log_overcrowding': log_overcrowding, 'log_SQBovercrowding': log_SQBovercrowding, 'log_SQBdependency': log_SQBdependency, 'log_SQBmeaned': log_SQBmeaned})
temp_df.head()
temp_df.describe()
# Converting all log-transformed variables to categorical variables according to their distribution as given in above temp_df.describe() cell
temp_df['log_meaneduc'] = pd.cut(temp_df['log_meaneduc'], [0.0, 1.945910, 2.268684, 2.525729, 3.637586], labels=[1, 2, 3, 4], include_lowest=True)
temp_df['log_overcrowding'] = pd.cut(temp_df['log_overcrowding'], [0.133531, 0.693147, 0.916291, 1.098612, 2.639057], labels=[1, 2, 3, 4], include_lowest=True)
temp_df['log_SQBovercrowding'] = pd.cut(temp_df['log_SQBovercrowding'], [0.020203, 0.693147, 1.178655, 1.609438, 5.135798], labels=[1, 2, 3, 4], include_lowest=True)
temp_df['log_SQBdependency'] = pd.cut(temp_df['log_SQBdependency'], [0.0, 0.105361, 0.367725, 1.021651, 4.174387], labels=[1, 2, 3, 4], include_lowest=True)
temp_df['log_SQBmeaned'] = pd.cut(temp_df['log_SQBmeaned'], [0.0, 3.610918, 4.332194, 4.892227, 7.222566], labels=[1, 2, 3, 4], include_lowest=True)
# Converting float variables to categorical variables introduces some nan values (don't know the reason), so I replaced them with the max values. Then, I replaced the
# original feature variables in all_df with these categorical variables. 
temp_df.fillna({'log_meaneduc': 2, 'log_overcrowding': 2, 'log_SQBovercrowding': 2, 'log_SQBdependency': 1, 'log_SQBmeaned': 4}, inplace=True)
all_df[['meaneduc', 'overcrowding', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']] = temp_df[['log_meaneduc', 'log_overcrowding', 'log_SQBovercrowding', 'log_SQBdependency', 'log_SQBmeaned']]
# Visualizing integer feature columns
int_flat = pd.melt(all_df, value_vars=int_cols)
g = sns.FacetGrid(int_flat, col='variable', col_wrap=6, sharex=False, sharey=False)
g = g.map(sns.countplot, 'value')
plt.xticks(rotation=90)
plt.show()
# Removing Id feature from list and visualizing object feature columns
del(object_cols[0])
object_flat = pd.melt(all_df, value_vars=object_cols)
g = sns.FacetGrid(object_flat, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.countplot, 'value')
plt.show()
# Encoding feature columns of 'object' dtype
le = LabelEncoder()
for col in object_cols:
    all_df[col] = le.fit_transform(all_df[col].values)
# Removing squared feature columns as these are having same distribution as the original variables
dup_cols = [col for col in all_df.columns if col[:3] == 'SQB']
all_df.drop(dup_cols, axis=1, inplace=True)
all_df.drop('agesq', axis=1, inplace=True)
all_df.shape
# Onehot-encoding feature variables having one prominent category
all_df['edjefe'] = (all_df['edjefe'] == all_df['edjefe'].max()) * 1
all_df['edjefa'] = (all_df['edjefa'] == all_df['edjefa'].max()) * 1

# Converting age and idhogar to float variables by scaling them with 0 mean and unit standard deviation
all_df['age'] = sc.fit_transform(all_df['age'].values.reshape((-1, 1)))
all_df['idhogar'] = sc.fit_transform(all_df['idhogar'].values.reshape((-1, 1)))
# Updating cat_cols list and then onehot-encoding all columns having more than 2 unique values

cat_cols = [col for col in int_cols if col not in [col for col in int_cols if col[:3]=='SQB']]
cat_cols.remove('agesq')
cat_cols.remove('age')

onehot_cols = [col for col in cat_cols if len(all_df[col].unique()) > 2]
onehot_arr = onehot.fit_transform(all_df[onehot_cols].values)
onehot_arr.shape
all_df.drop(onehot_cols, axis=1, inplace=True)
all_df.shape
# Dividing the whole dataframe into train and test dataframes
train_df = all_df[all_df['Target'].notnull()]
test_df = all_df[all_df['Target'].isnull()]
print (train_df.shape, test_df.shape)
# We have to reduce the target value of each class by 1, otherwise XgBoost thinks its training on 5 classes, since highest class is 4. We will undo this change
# after prediction
train_df['Target'] = train_df['Target'].apply(lambda x: x-1)
# Creating train array and test array from train dataframe and test dataframe and also concatenating onehot array
tr_cols = [col for col in train_df.columns if col not in ['Id', 'Target']]
X_train = train_df[tr_cols].values
X_train = np.concatenate((X_train, onehot_arr[:9557, :]), axis=1)
y_train = train_df['Target'].values

te_cols = [col for col in test_df.columns if col not in ['Id', 'Target']]
X_test = test_df[te_cols].values
X_test = np.concatenate((X_test, onehot_arr[9557:, :]), axis=1)
# Reducing dimensionality of complete array to 2-dimensions(for visualizing) using t-distributed Stochastic Neighbor Embedding algorithm, so that we can visualize how
# different models are predicting on test set.
all_arr = np.concatenate((X_train, X_test))
tsne = TSNE(n_components=2)
all_tsne_arr = tsne.fit_transform(all_arr)
all_tsne_arr.shape
# Declaring class weights as the 4 classes are imbalanced
class_weights = compute_class_weight('balanced', np.sort(train_df['Target'].unique()), train_df['Target'].values)
# Initializing CatBoost classifier, fitting and then predicting
cat_model = CatBoostClassifier(iterations=500, learning_rate=0.3, depth=5, loss_function='MultiClass', classes_count=4, logging_level='Silent', l2_leaf_reg=2, thread_count=4, class_weights=class_weights)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
cat_preds = cat_preds.reshape((-1,)).astype(int)
# Initializing Random Forest classifier, fitting and then predicting
rfc_clf = RandomForestClassifier(n_estimators=70, max_depth=5, max_features=0.8, n_jobs=4, class_weight='balanced')
rfc_clf.fit(X_train, y_train)
rfc_preds = rfc_clf.predict(X_test).astype(int)
# Initializing Adam Boost classifier, fitting and then predicting
ada_clf = AdaBoostClassifier(n_estimators=70, learning_rate=0.3)
ada_clf.fit(X_train, y_train, sample_weight=[class_weights[int(y_train[i])] for i in range(y_train.shape[0])])
ada_preds = ada_clf.predict(X_test).astype(int)
# Initializing Bernoulli naive-bayes classifier, fitting and then predicting
bernoulli_clf = BernoulliNB()
bernoulli_clf.fit(X_train, y_train, sample_weight=[class_weights[int(y_train[i])] for i in range(y_train.shape[0])])
bernoulli_preds = bernoulli_clf.predict(X_test).astype(int)
# Initializing Gaussian naive-bayes classifier, fitting and then predicting
gaussian_clf = GaussianNB()
gaussian_clf.fit(X_train, y_train, sample_weight=[class_weights[int(y_train[i])] for i in range(y_train.shape[0])])
gaussian_preds = gaussian_clf.predict(X_test).astype(int)
# Initializing KNN classifier, fitting and then predicting
knn_clf = KNeighborsClassifier(n_neighbors=8, weights='uniform', n_jobs=4)
knn_clf.fit(X_train, y_train)
knn_preds = knn_clf.predict(X_test).astype(int)
# Initializing Multilayer Perceptron, fitting and then predicting
mlp_clf = MLPClassifier(hidden_layer_sizes=(50), batch_size=50, learning_rate='constant', learning_rate_init=0.0005, early_stopping=True)
mlp_clf.fit(X_train, y_train)
mlp_preds = mlp_clf.predict(X_test).astype(int)
# Visualizing all predictions from above models using a scatter-plot so that we can also see the differences between their predictions
plt.figure(figsize=(30, 30))

plt.subplot(421)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=cat_preds)
plt.colorbar()
plt.title('Catboost predictions')
plt.grid(True)

plt.subplot(422)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=rfc_preds)
plt.colorbar()
plt.title('RandomForest predictions')
plt.grid(True)

plt.subplot(423)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=ada_preds)
plt.colorbar()
plt.title('Adaboost predictions')
plt.grid(True)

plt.subplot(424)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=bernoulli_preds)
plt.colorbar()
plt.title('Bernoulli predicitons')
plt.grid(True)

plt.subplot(425)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=gaussian_preds)
plt.colorbar()
plt.title('Gaussian predictions')
plt.grid(True)

plt.subplot(426)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=knn_preds)
plt.colorbar()
plt.title('KNN predictions')
plt.grid(True)

plt.subplot(427)
plt.scatter(all_tsne_arr[9557:, 0], all_tsne_arr[9557:, 1], c=mlp_preds)
plt.colorbar()
plt.title('Multi-layer Perceptron predictions')
plt.grid(True)

plt.show()
# Combining all predictions and creating a heatplot of their correlations
all_preds = np.concatenate((cat_preds.reshape((-1, 1)), rfc_preds.reshape((-1, 1))), axis=1)
all_preds = np.concatenate((all_preds, ada_preds.reshape((-1, 1))), axis=1)
all_preds = np.concatenate((all_preds, bernoulli_preds.reshape((-1, 1))), axis=1)
all_preds = np.concatenate((all_preds, gaussian_preds.reshape((-1, 1))), axis=1)
all_preds = np.concatenate((all_preds, knn_preds.reshape((-1, 1))), axis=1)
all_preds = np.concatenate((all_preds, mlp_preds.reshape((-1, 1))), axis=1)

all_preds_df = pd.DataFrame(all_preds, columns=['cat_preds', 'rfc_preds', 'ada_preds', 'bernoulli_preds', 'gaussian_preds', 'knn_preds', 'mlp_preds'])
sns.heatmap(all_preds_df.corr())
plt.show()
# Visualizing and comparing the freqencies of predicted classes of all models
all_preds_flat = pd.melt(all_preds_df)
g = sns.FacetGrid(all_preds_flat, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.countplot, 'value')
plt.show()
# Creating level1_train dataset
level1_train = np.zeros((X_train.shape[0], 28))
skf = StratifiedKFold(n_splits=5)

for tr_idx, te_idx in skf.split(X_train, y_train):
    
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_te, y_te = X_train[te_idx], y_train[te_idx]
    
    cat_model.fit(X_tr, y_tr)
    cat_preds = cat_model.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7] = cat_preds[:, i]
    
    rfc_clf.fit(X_tr, y_tr)
    rfc_preds = rfc_clf.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7+1] = rfc_preds[:, i]
    
    ada_clf.fit(X_tr, y_tr)
    ada_preds = ada_clf.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7+2] = ada_preds[:, i]
    
    bernoulli_clf.fit(X_tr, y_tr)
    bernoulli_preds = bernoulli_clf.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7+3] = bernoulli_preds[:, i]
    
    gaussian_clf.fit(X_tr, y_tr)
    gaussian_preds = gaussian_clf.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7+4] = gaussian_preds[:, i]
    
    knn_clf.fit(X_tr, y_tr)
    knn_preds = knn_clf.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7+5] = knn_preds[:, i]

    mlp_clf.fit(X_tr, y_tr)
    mlp_preds = mlp_clf.predict_proba(X_te)
    for i in range(4):
        level1_train[te_idx, i*7+6] = mlp_preds[:, i]
# Creating level1_test dataset
level1_test = np.zeros((X_test.shape[0], 28))

cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7] = cat_preds[:, i]

rfc_clf.fit(X_train, y_train)
rfc_preds = rfc_clf.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7+1] = rfc_preds[:, i]

ada_clf.fit(X_train, y_train)
ada_preds = ada_clf.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7+2] = ada_preds[:, i]

bernoulli_clf.fit(X_train, y_train)
bernoulli_preds = bernoulli_clf.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7+3] = bernoulli_preds[:, i]
    
gaussian_clf.fit(X_train, y_train)
gaussian_preds = gaussian_clf.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7+4] = gaussian_preds[:, i]
    
knn_clf.fit(X_train, y_train)
knn_preds = knn_clf.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7+5] = knn_preds[:, i]

mlp_clf.fit(X_train, y_train)
mlp_preds = mlp_clf.predict_proba(X_test)
for i in range(4):
    level1_test[:, i*7+6] = mlp_preds[:, i]
# Training a meta classifier on level1_train dataset and making predictions on level1_test dataset
meta_clf = RidgeClassifier(normalize=True, class_weight='balanced')
meta_clf.fit(level1_train, y_train)
meta_preds = meta_clf.predict(level1_test).astype(int)

meta_subm = pd.read_csv('../input/sample_submission.csv')
meta_subm['Target'] = meta_preds

sns.countplot(meta_subm['Target'])
plt.show()
meta_subm['Target'] = meta_subm['Target'].apply(lambda x: x+1)
meta_subm.to_csv('Stack_1.csv', index=False)