# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

df_train.head()
df_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

df_test.head()
df_train.shape , df_test.shape
df_train.describe()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x='target',data=df_train)
df_train.info()
num_bins = 5

plt.hist(df_test['var_0'], num_bins, density=1, facecolor='LightSeaGreen', alpha=0.5)

plt.show()
X = df_train.iloc[:,2:202]

y = df_train.iloc[:,1]
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

%matplotlib inline

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split,  cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)
X_train.shape
y_train.shape
classifier = LogisticRegression(random_state=200)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

confusion_matrix
auc = roc_auc_score(y_test, y_pred)

print('AUC: %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.show()
y_pred = classifier.predict(df_test)
sub_df = pd.DataFrame({'ID_code':df_test.ID_code.values})

sub_df['target'] = y_pred

sub_df.to_csv('submission.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
first_knn = KNeighborsClassifier()
first_tree = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
tree_params = {'max_depth': np.arange(1,11), 'max_features' : range(4, 1)}
tree_params = {'max_depth': range(1,11),

'max_features': range(1,10)}
tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)
tree_grid.best_score_, tree_grid.best_params_
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
knn_params = {'knn__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_params,

cv=5, n_jobs=-1,

verbose=True)
from sklearn.tree import export_graphviz
export_graphviz(tree_grid.best_estimator_, feature_names=X.columns, 

out_file='tree.dot', filled=True,)
!dot -Tpng tree.dot -o tree.png