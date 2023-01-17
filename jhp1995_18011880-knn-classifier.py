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
train_df = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')
test_df = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')
submission_df = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')
train_df.drop('Unnamed: 0', axis=1, inplace=True)
train_df.head()
import seaborn as sns # seaborn 불러옴
sns.set() # seaborn style로 그림 그리기
sns.pairplot(train_df,hue='8') # size는 그림 크기
y_train = train_df['8']
X_train = train_df.drop(['2','8'], axis=1)
X_train.head()
y_train.head()
test_df.drop(['Unnamed: 0', '2','8'], axis=1, inplace=True)
test_df.head()
submission_df.head()
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  #KNN 불러오기
knn_clf = KNeighborsClassifier(weights='distance')
parameters = {'n_neighbors':[40, 42, 44, 46],
             'leaf_size':[1, 2, 3],
             'p':[1, 2, 3, 4, 5]}
grid_knn_clf = GridSearchCV(knn_clf, param_grid=parameters, scoring='accuracy', cv = 5)
grid_knn_clf.fit(X_train, y_train)
print(grid_knn_clf.best_params_)
print(grid_knn_clf.best_score_)
best_knn_clf = grid_knn_clf.best_estimator_
predictions = best_knn_clf.predict(test_df)
for i in range(len(predictions)):
    submission_df['Label'][i] = predictions[i]
submission_df=submission_df.astype(np.int32)
submission_df.to_csv('submit.csv', mode='w', header= True, index= False)