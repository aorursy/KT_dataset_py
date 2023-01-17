# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
from scipy.stats import pointbiserialr
import seaborn as sns
data = pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
test_data = pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")
Y = data['flag']
data = data.drop('flag',axis = 1)
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
data = pd.DataFrame(data)
data.columns = test_data.columns
scaler.fit(test_data)
test_data = pd.DataFrame(scaler.transform(test_data))
test_data.columns  = data.columns
data['flag'] = Y
[pointbiserialr(data['flag'],data[col]) for col in data.columns]
corr = data.drop('flag',axis = 1).corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
data = data.drop(['positionBack','refPositionBack', 'refVelocityBack','refPositionFront', 'refVelocityFront','velocityFront'], axis = 1)
test_data = test_data.drop(['positionBack','refPositionBack', 'refVelocityBack','refPositionFront', 'refVelocityFront','velocityFront'], axis = 1)
X_train, X_val, Y_train, Y_val = train_test_split(data.drop(['flag','timeindex'],axis = 1), data['flag'], test_size=0.33, random_state=1234)
model_rf = RandomForestClassifier(n_estimators = 170, random_state = 42,criterion='gini', max_depth=60, max_features='auto')
model_rf.fit(X_train,Y_train)
print(fbeta_score(model_rf.predict(X_train), Y_train,0.5))
print(fbeta_score(model_rf.predict(X_val), Y_val,0.5))
rf_values = model_rf.predict(X_val)
output = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
output['flag'] = model_rf.predict(test_data.drop('timeindex',axis = 1))
output.to_csv('submission_rf_2.csv', index=False)
model_dt = DecisionTreeClassifier(max_leaf_nodes = 310, random_state = 1234)
model_dt.fit(X_train,Y_train)
print(fbeta_score(model_dt.predict(X_train), Y_train,0.5))
print(fbeta_score(model_dt.predict(X_val), Y_val,0.5))
dt_values = model_dt.predict(X_val)
output = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
output['flag'] = model_dt.predict(test_data.drop('timeindex',axis = 1))
output.to_csv('submission_dt.csv', index=False)
model_voting = VotingClassifier(estimators=[('RandomForest', model_rf), ('Decision_Tree', model_dt)], voting='soft')
model_voting.fit(X_train, Y_train)
print(fbeta_score(model_voting.predict(X_train), Y_train,0.5))
print(fbeta_score(model_voting.predict(X_val), Y_val,0.5))
output = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
output['flag'] = model_dt.predict(test_data.drop('timeindex',axis = 1))
output.to_csv('submission_cv.csv', index=False)