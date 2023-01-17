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
X_train = pd.read_csv("../input/X_train.csv")

y_train = pd.read_csv("../input/y_train.csv")

X_test = pd.read_csv("../input/X_test.csv")



X_train.head()
X_test.head()
y_train.head()
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(15, 5))

sns.countplot(y_train['surface'])

plt.title('Target distribution', size=20)

plt.show()
missing_data = X_train.isnull().sum()

print ("Missing Data in Training set")

missing_data.tail()
missing_data = X_train.isnull().sum()

print ("Missing Data in Test set")

missing_data.tail()
def feature_engineering(data):

    

    df = pd.DataFrame()

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +

                             data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +

                             data['linear_acceleration_Z'])**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 +

                             data['orientation_Z'])**0.5

   

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    

    for col in data.columns:

        if col in ['row_id','series_id','measurement_number']:

            continue

        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()

        df[col + '_median'] = data.groupby(['series_id'])[col].median()

        df[col + '_max'] = data.groupby(['series_id'])[col].max()

        df[col + '_min'] = data.groupby(['series_id'])[col].min()

        df[col + '_std'] = data.groupby(['series_id'])[col].std()

        df[col + '_range'] = df[col + '_max'] - df[col + '_min']

        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']

        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2

    return df
%%time

X_train = feature_engineering(X_train)

X_test = feature_engineering(X_test)

print(X_train.shape)

y_train.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



y_train['surface']= le.fit_transform(y_train['surface'])

y_train.head()
X_train.fillna(0, inplace = True)

X_test.fillna(0, inplace = True)

X_train.replace(-np.inf,0,inplace=True)

X_train.replace(np.inf,0,inplace=True)

X_test.replace(-np.inf,0,inplace=True)

X_test.replace(np.inf,0,inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import StratifiedKFold



def k_folds(X, y, X_test, k):

    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=20)

    y_test = np.zeros((X_test.shape[0], 9))

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  RandomForestClassifier(n_estimators = 100, n_jobs = -1)

        clf.fit(X_train.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = clf.predict(X.iloc[val_idx])

        y_test += clf.predict_proba(X_test) / folds.n_splits

        score += clf.score(X.iloc[val_idx], y[val_idx])

#         print('Fold: {} score: {}'.format(i,clf.score(X.iloc[val_idx], y[val_idx])))

    print('Avg Accuracy', score / folds.n_splits) 

        

    return y_oof, y_test 
y_oof, y_test = k_folds(X_train, y_train['surface'], X_test, k= 10)
confusion_matrix(y_oof,y_train['surface'])
y_test = np.argmax(y_test, axis=1)

submission = pd.read_csv(os.path.join("../input/",'sample_submission.csv'))

submission['surface'] = le.inverse_transform(y_test)

submission.to_csv('submission.csv', index=False)

submission.head(10)