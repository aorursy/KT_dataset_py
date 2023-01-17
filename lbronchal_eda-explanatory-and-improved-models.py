import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_demographic = pd.read_csv('../input/demographic_info.csv')
data_demographic
data_eeg = pd.read_csv('../input/EEG_data.csv')
data_eeg.head()
data_eeg.info()
data_eeg['SubjectID'] = data_eeg['SubjectID'].astype(int)
data_eeg['VideoID'] = data_eeg['VideoID'].astype(int)
data_eeg['predefinedlabel'] = data_eeg['predefinedlabel'].astype(int)
data_eeg['user-definedlabeln'] = data_eeg['user-definedlabeln'].astype(int)
data_eeg.iloc[:, 2:].describe()
data_eeg['user-definedlabeln'].value_counts()
data_resume = data_eeg.groupby(['SubjectID', 'VideoID'])['user-definedlabeln'].agg(lambda x: sum(x) > 0).unstack("VideoID")
data_resume
fig = plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
data_resume.apply(sum).plot(kind='bar', title='Number of subjets surprised by a video')
plt.subplot(1, 2, 2)
data_resume.apply(sum, axis=1).plot(kind='bar', title="Number of videos that surprised a subject")
plt.show()
data_user1_video1 = data_eeg.query('SubjectID==0 & VideoID==0')
len(data_user1_video1)
features = ['Attention', 'Mediation', 'Raw', 'Delta',
            'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
data_user1_video1[features].plot(figsize=(18,6))
plt.show()
(data_user1_video1['user-definedlabeln']!=0).sum()
data_eeg.groupby(['SubjectID', 'VideoID']).size().loc[(slice(None), 2)].plot(kind='bar', figsize=(12,6))
plt.title("Number of rows per user for video #3")
plt.ylabel("Number of rows")
plt.show()
data_eeg.isnull().any().sum()
data_eeg['Attention'].plot(figsize=(18,6))
plt.show()
data_eeg['Mediation'].plot(figsize=(18,6))
plt.show()
data_eeg['Raw'].plot(figsize=(18,6))
plt.show()
data_eeg.groupby(['SubjectID', 'VideoID']).filter(lambda x: x['Attention'].sum()==0).groupby(['SubjectID', 'VideoID']).size()
data_eeg.groupby(['SubjectID', 'VideoID']).filter(lambda x: x['Mediation'].sum()==0).groupby(['SubjectID', 'VideoID']).size()
data = data_eeg.query('(SubjectID != 6) & (SubjectID != 3 | VideoID !=3)')
len(data), len(data_eeg)
data.reset_index()['Attention'].plot(figsize=(18,6))
plt.show()
corr = data[features].corr()
corr
import seaborn as sns
plt.figure(figsize = (12, 12))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="RdBu_r")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
seed = 123
X = data[features]
y = data['user-definedlabeln']
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)
model_lr = LogisticRegression(random_state=seed)
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
print("Test Accuracy: {:.5f}".format(accuracy_score(y_test, pred_lr)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier(max_depth=2)
model_tree.fit(X_train, y_train)
pred_tree = model_tree.predict(X_test)

print("Test Accuracy: {:.5f}".format(accuracy_score(y_test, pred_tree)))
import graphviz 
from sklearn.tree import export_graphviz
tree_view = export_graphviz(model_tree, 
                            out_file=None, 
                            feature_names = features,
                            class_names = ['No confused', 'Confused'])  
tree1viz = graphviz.Source(tree_view)
tree1viz
import xgboost as xgb
model_xgb = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=seed)
model_xgb.fit(X_train, y_train)
data_dmatrix = xgb.DMatrix(data=X.values, label=y.values)

# Create the parameter dictionary: params
#params = {"objective":"reg:logistic", "max_depth":5, "eta":0.1, "n_estimators":1000, "colsample_bytree": 0.7, "learning_rate": 0.1}
params = {}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=seed)

print("Test Accuracy: {:.5f}".format(((1-cv_results["test-error-mean"]).iloc[-1])))
data_demographic.columns = ['subject ID', 'age', 'ethnicity', 'gender']
data.head()
data_extended = data.merge(data_demographic, left_on="SubjectID", right_on="subject ID")
data_extended.head()
data_extended['ethnicity'] = data_extended['ethnicity'].astype("category").cat.codes
data_extended['gender'] = data_extended['gender'].astype("category").cat.codes
features_extra = features + ['age', 'ethnicity', 'gender']
data_dmatrix = xgb.DMatrix(data=data_extended[features_extra].values, label=y.values)

# Create the parameter dictionary: params
#params = {"objective":"reg:logistic", "max_depth":5, "eta":0.1, "n_estimators":1000, "colsample_bytree": 0.7, params = {"learning_rate": 0.1}}
params = {}
# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=seed)

# Print the accuracy
print("Test Accuracy: {:.5f}".format(((1-cv_results["test-error-mean"]).iloc[-1])))