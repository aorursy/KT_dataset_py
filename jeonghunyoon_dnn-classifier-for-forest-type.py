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
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
display(train_set.head())
display(train_set.describe())
display(train_set.keys())
display(len(train_set.keys()))
# Checking binary column
soil_cols = ['Soil_Type' + str(i) for i in range(1, 41)]
wilder_cols = ['Wilderness_Area' + str(i) for i in range(1, 5)]

# If sum : 15120 => data is OK!
display(train_set[soil_cols].sum(axis=1).sum(axis=0))
display(train_set[wilder_cols].sum(axis=1).sum(axis=0))
# categorical variable
cate_vars = soil_cols[:]
cate_vars.extend(wilder_cols)

# continuous variable
cont_vars = list(train_set.keys())
cont_vars.remove('Id')
cont_vars.remove('Cover_Type')
cont_vars = [var for var in cont_vars if var not in cate_vars]

print(cate_vars)
print(cont_vars)
# How about using this features directly? (Not using the scaling and normalization)
fig = plt.figure()
fig.set_size_inches(35, 35)
sns.set(font_scale=2)

# Delete 'Id' and change cover type to dummy variables
cont_var_train_set = train_set.drop('Id', axis=1).drop(cate_vars, axis=1)

# Categorical feature : cannot using correlation directly.
cont_var_train_set_dum = pd.get_dummies(cont_var_train_set, columns=['Cover_Type'])

correlation = cont_var_train_set_dum.corr()
sns.heatmap(correlation, cmap='viridis', annot=True, linewidths=3)
from sklearn.preprocessing import StandardScaler
# After feature scailing : Actually, it is same to above correlation
scaled_feat = cont_var_train_set_dum.iloc[:, :-7]
dummy_labels = cont_var_train_set_dum.iloc[:, -7:]
# using scaler
scaler = StandardScaler()
scaler.fit(scaled_feat)
scaled_feat = scaler.transform(scaled_feat)

scaled_feat = pd.DataFrame(scaled_feat, columns=cont_vars)
scaled_feat.head()
fig = plt.figure()
fig.set_size_inches(35, 35)

correlation2 = pd.concat([scaled_feat, dummy_labels], axis=1).corr()
sns.heatmap(correlation2, cmap='viridis', annot=True, linewidths=3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Spliting the datasets
features = pd.concat([scaled_feat, train_set[cate_vars]], axis=1)
features.head()
x_train, x_test, y_train, y_test = train_test_split(features, train_set['Cover_Type'], random_state=20190425, test_size=0.3)
rf_model = RandomForestClassifier(max_depth=7, n_estimators=300)
rf_model.fit(x_train, y_train)
# Predicting naively
pred = rf_model.predict(x_test)

display(accuracy_score(y_test, pred))
display(classification_report(y_test, pred))
# See the importance of features
importances = rf_model.feature_importances_
indices = np.argsort(importances)

fig = plt.figure()
fig.set_size_inches(20, 20)
sns.set(font_scale=1.5)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features.keys()[indices])
plt.xlabel('Relative Importance')
# dimensional reduction
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=None, random_state=20180425)
pca.fit(features)
pca_var = pca.explained_variance_ratio_

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax1, ax2 = ax.flatten()

ax1.plot(pca_var)
ax2.plot(np.cumsum(pca_var))
train_set.head()
for idx, row in train_set.iterrows():
    for i in range(1, 5):
        if row['Wilderness_Area' + str(i)] == 1:
            train_set.loc[idx, 'Wilderness_Area'] = i
    for i in range(1, 41):
        if row['Soil_Type' + str(i)] == 1:
            train_set.loc[idx, 'Soil_Type'] = i
train_set.head()
wilderness_area_col = train_set['Wilderness_Area'].astype(int)
soil_type_col = train_set['Soil_Type'].astype(int)

display(wilderness_area_col.head())
display(soil_type_col.head())
# train_set = train_set.drop(['Soil_Type'+str(idx) for idx in range(1, 41)], axis=1)
# train_set = train_set.drop(['Wilderness_Area'+str(idx) for idx in range(1, 5)], axis=1)
import scipy.stats as ss
# get confusion matrix manually
confusions = []
for soil in range(1, 41):
    for cover in range(1, 8):
        cond = train_set[(train_set['Soil_Type'] == soil) & (train_set['Cover_Type'] == cover)]
        confusions.append(cond.count()['Soil_Type'])
confusion_matrix = np.array(confusions).reshape(40, 7)
confusion_matrix = confusion_matrix[confusion_matrix.sum(axis=1) > 0]
# cramers v stat 1
def get_cramers_stat(confusion_matrix):
    confusion_matrix = confusion_matrix
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    cramers_stat = np.sqrt(phi2 / (min(confusion_matrix.shape)-1))
    return cramers_stat

soil_type_result_1 = get_cramers_stat(confusion_matrix)
print(soil_type_result_1)
confusion_matrix = pd.crosstab(train_set['Soil_Type'], train_set['Cover_Type'])
confusion_matrix = np.array(confusion_matrix)
soil_type_result_2 = get_cramers_stat(confusion_matrix)
print(soil_type_result_2)
confusion = []
for wilderness in range(1, 5):
    for cover in range(1, 8):
        cond = train_set[(train_set['Wilderness_Area'] == wilderness) & (train_set['Cover_Type'] == cover)]
        confusion.append(cond.count()['Wilderness_Area'])
confusion_matrix = np.array(confusion).reshape(4, 7)
wilderness_area_result_1 = get_cramers_stat(confusion_matrix) 
print(wilderness_area_result_1)
confusion_matrix = pd.crosstab(train_set['Wilderness_Area'], train_set['Cover_Type'])
confusion_matrix = np.array(confusion_matrix)
wilderness_area_result_2 = get_cramers_stat(confusion_matrix)
print(wilderness_area_result_2)
cate_vars_1 = ['Wilderness_Area', 'Soil_Type']
input_features = pd.concat([scaled_feat, wilderness_area_col, soil_type_col], axis=1)
labels = train_set['Cover_Type']

display(input_features.head())
display(labels.head())
x_train, x_test, y_train, y_test = train_test_split(input_features, labels, random_state=20190501, test_size=0.3)
import tensorflow as tf
wilderness_area_cate_list = list(set(input_features['Wilderness_Area']))
soil_type_cate_list = list(set(input_features['Soil_Type']))

wilderness_area_cols = tf.feature_column.categorical_column_with_vocabulary_list(
    'Wilderness_Area', wilderness_area_cate_list
)

soil_type_cols = tf.feature_column.categorical_column_with_vocabulary_list(
    'Soil_Type', soil_type_cate_list
)

embed_wilderness_area_cols = tf.feature_column.embedding_column(
    categorical_column=wilderness_area_cols,
    dimension = 5
#     dimension = round(len(wilderness_area_cate_list) ** 0.25)
)

embed_soil_type_cols = tf.feature_column.embedding_column(
    categorical_column=soil_type_cols,
    dimension = 5
#     dimension = round(len(soil_type_cate_list) ** 0.25)
)
test_set_rf = test_set.copy()
test_set_rf_cont = test_set_rf[cont_vars]

scaler.fit(test_set_rf_cont)
test_set_rf_cont = scaler.transform(test_set_rf_cont)
test_set_rf_cont = pd.DataFrame(test_set_rf_cont, columns=cont_vars)
test_set_rf_cate = test_set_rf[cate_vars]

scaled_test_set_rf = pd.concat([test_set_rf_cont, test_set_rf_cate], axis=1)
scaled_test_set_rf.head()
rf_pred = rf_model.predict(scaled_test_set_rf)
rf_result = pd.concat([test_set['Id'], pd.DataFrame({'Cover_Type': rf_pred})], axis=1)
rf_result.to_csv("rf_submission.csv", index=False)
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x = x_train,
    y = y_train,
    num_epochs = 30,
    batch_size = 50,
    shuffle=True
)

eval_input_fn = tf.estimator.inputs.pandas_input_fn(
    x = x_test,
    y = y_test,
    num_epochs = 1,
    shuffle = False
)
tf_features = []

# used standard scaler in sklearn
[tf_features.append(tf.feature_column.numeric_column(col)) 
 for col in cont_vars]

tf_features.extend([embed_wilderness_area_cols, embed_soil_type_cols])
tf_features
estimator = tf.estimator.DNNClassifier(
    feature_columns = tf_features,
    hidden_units = [1024, 512, 256],
    n_classes = 8,
    optimizer = tf.train.AdamOptimizer()
)
estimator.train(input_fn=train_input_fn)
estimator.evaluate(input_fn=eval_input_fn)
# Test set setting
test_set_copy = test_set.copy()

# categorical data transfer
for idx, row in test_set_copy.iterrows():
    for i in range(1, 5):
        if row['Wilderness_Area' + str(i)] == 1:
            test_set_copy.loc[idx, 'Wilderness_Area'] = i
    for i in range(1, 41):
        if row['Soil_Type' + str(i)] == 1:
            test_set_copy.loc[idx, 'Soil_Type'] = i
# 1. scaling the continous features
test_cont_feat = test_set_copy[cont_vars]
scaler.fit(test_cont_feat)
test_scaled_cont_feat = scaler.transform(test_cont_feat)
test_scaled_cont_feat = pd.DataFrame(test_scaled_cont_feat, columns=cont_vars)

# 2. categorical features
test_cate_feat = test_set_copy[cate_vars_1].astype(int)

# 3. concat
test_input_features = pd.concat([test_scaled_cont_feat, test_cate_feat], axis=1)
display(test_cont_feat.head())
display(test_scaled_cont_feat.head())
display(test_input_features.head())
# 3. prediction input function
pred_input_fn = tf.estimator.inputs.pandas_input_fn(
    x = test_input_features,
    num_epochs = 1,
    shuffle = False
)

predictions = list(estimator.predict(pred_input_fn))
predicted_classes = [int(pred['classes']) for pred in predictions]
result = predicted_classes[:]
result = pd.concat([test_set['Id'], pd.DataFrame({'Cover_Type': result})], axis=1)
result.head()
result.to_csv("submission.csv", index=False)