import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly import tools # for subplots in plotly
import plotly.graph_objs as go # create object to visualisize data
from plotly.offline import init_notebook_mode, iplot # display plot
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
plt.rcParams['figure.figsize'] = 16,8
plt.style.use('ggplot')
#raw_data_train = pd.read_csv("../input/ks-projects-201612.csv",skiprows=10)
raw_data_train = pd.read_csv("../input/ks-projects-201801.csv")
raw_data_train.head()
X = raw_data_train[['usd_pledged_real', 'usd_goal_real']]
y = raw_data_train['state']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5, random_state=42, stratify=y_train)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print(f"Score of model in training set: {accuracy_score(y_train,dtc.predict(X_train))}")
print(f"Score of model in valid set: {accuracy_score(y_valid,dtc.predict(X_valid))}")
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred_valid = dtc.predict(X_valid)
mat = confusion_matrix(y_valid, y_pred_valid)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,\
            xticklabels=y_valid.unique(), yticklabels=y_valid.unique())
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.model_selection import GridSearchCV
param = {
'max_depth' : [3, 5, 10, 15, 20]
}
dtc = DecisionTreeClassifier()
tree = GridSearchCV(dtc, param)
tree.fit(X_train, y_train)
scores_train = tree.cv_results_['mean_train_score']
scores_valid = tree.cv_results_['mean_test_score']
depth = param['max_depth']
plt.plot(depth, scores_train)
plt.plot(depth, scores_valid)
plt.legend(['train', 'test'])
plt.xlabel("depth tree")
plt.ylabel("scores");
model10 = tree.best_estimator_
model10
model10.fit(X_train, y_train)
y_pred_valid = model10.predict(X_valid)
mat = confusion_matrix(y_valid, y_pred_valid)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,\
            xticklabels=y_valid.unique(), yticklabels=y_valid.unique())
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10)
rf
rf.fit(X_train, y_train)
y_pred_valid = rf.predict(X_valid)
mat = confusion_matrix(y_valid, y_pred_valid)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,\
            xticklabels=y_valid.unique(), yticklabels=y_valid.unique())
plt.xlabel('true label')
plt.ylabel('predicted label');
rf20 = RandomForestClassifier(n_estimators=20, max_depth=10)
rf20.fit(X_train, y_train)
y_pred_valid = rf20.predict(X_valid)
mat = confusion_matrix(y_valid, y_pred_valid)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,\
            xticklabels=y_valid.unique(), yticklabels=y_valid.unique())
plt.xlabel('true label')
plt.ylabel('predicted label');
y_pred = [rf20.estimators_[i].predict(X_test) for i in range(len(rf20.estimators_))]
y_pred = np.array(y_pred)
import scipy.stats as st
pred_mode = st.mode(y_pred).mode
pred_mode
pred_count = st.mode(y_pred).count
pred_count
counts = pd.DataFrame({'pred': pred_mode[0], 'count': pred_count[0]})
counts.head()
# Or we can use squeeze to remove useless dimensions
freqs = pd.DataFrame({'pred': np.squeeze(pred_mode), 'frequency': np.squeeze(pred_count) / len(rf20.estimators_) * 100})
freqs.head()
freqs.groupby(['frequency','pred'])['pred'].agg('count').unstack()
