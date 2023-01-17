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
# import essentials
import numpy as np
import pandas as pd

# set randomness for reproducability
seed = 42
np.random.seed(seed)
# get our data
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
# give it a quick overview
print(data.describe())
# check data types of our data
print('\ndata types of our dataset')
print(str(data.dtypes) + '\n')
# grab our column names to iterate over
columns = data.keys()
# changing almost all to categorical variables
# will make a few exceptions
column_exceptions = ['class','bruises']
newData = pd.DataFrame()
newData['class'] = (data['class'] == 'p')
newData['class'] = newData['class'].astype(int)
newData['bruises'] = data['bruises'] == 't'
data.drop(columns=['class','bruises'])
dummy_data = pd.get_dummies(data,drop_first=True, dtype='int')
newData = pd.concat([newData['class'],newData['bruises'],dummy_data],axis=1)
data = newData
# check dtypes again
print('\nCleaned data types')
print(data.dtypes)

from sklearn.model_selection import train_test_split

data_independent = data.iloc[:, 1:]  # X
data_dependent = data.iloc[:, 0] # y
X_train, X_test, y_train, y_test = train_test_split(
    data_independent,data_dependent.values,shuffle=True,
    random_state=seed, test_size=.2, stratify=data_dependent)
# import scoring metrics for evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# our cross validation strategy
def log_loss(model, X=X_train, y=y_train, _scoring='neg_log_loss') :
    _kfold = KFold(n_splits=5)
    _score = -cross_val_score(model, X, y, cv=_kfold, scoring=_scoring)
    return _score
# and the ratio that we predicted correctly
def final_accuracy(model,_X_train=X_train,_y_train=y_train,
                   _X_test=X_test,_y_test=y_test) :
    model.fit(_X_train,_y_train)
    _y_hat = model.predict(_X_test)
    _final_score = np.sum(_y_hat == _y_test) / len(_y_test)
    return _final_score
# now make linear estimations and evaluate if we need
# a more complex model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print('\nLoading and scoring models...')
logit = LogisticRegression(random_state=seed)
svc = SVC(probability=True,random_state=seed,kernel='rbf')
forest = RandomForestClassifier(random_state=seed)
# using log loss, what score do we get on our test set?
score = log_loss(logit)
print('\nLogit score: {:.4f} ({:.4f})'
      .format(score.mean(),score.std()))
score = log_loss(svc)
print('\nSVC score: {:.4f} ({:.4f})'
      .format(score.mean(),score.std()))
score = log_loss(forest)
print('\nRandom Forest score: {:.4f} ({:.4f})'
      .format(score.mean(),score.std()))

accuracy = final_accuracy(logit)
print('Logit final accuracy: {:.4f}'.format(accuracy))
accuracy = final_accuracy(svc)
print('\nSVC final accuracy: {:.4f}'.format(accuracy))
accuracy = final_accuracy(forest)
print('\nRandom Forest final accuracy: {:.4f}'.format(accuracy))
logit.fit(X_train,y_train)
predictions = logit.predict(X_test)

from sklearn.decomposition import PCA
# create our PCA and fit it to the data
pca = PCA(n_components=2, random_state=seed)
pca.fit(X_test)
X_test_pca = pca.transform(X_test)
# create our PCA dataframes
pca_df = pd.DataFrame(data=X_test_pca, columns=['PCA 1', 'PCA 2'])
y_test_series = pd.DataFrame(y_test, columns=['target'])
final_df = pd.concat([pca_df, y_test_series],axis=1)
# import our plotting methods
import matplotlib.pyplot as plt

# create our figure
fig :plt.Figure = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title('PCA Graph', fontsize=20)
targets = [0, 1]
colors = ['b','r']

for target, color in zip(targets, colors) :
    kept_indicies = predictions == target
    ax.scatter(final_df.loc[kept_indicies, 'PCA 1'],
               final_df.loc[kept_indicies, 'PCA 2'],
               c = color,
               s= 50)
ax.legend(['Non-Poisonous','Poisonous'])
ax.grid()
plt.show()