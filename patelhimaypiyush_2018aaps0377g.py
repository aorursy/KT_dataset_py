# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from imblearn.over_sampling import SMOTE



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df.head()
df.drop(['id'], axis=1, inplace=True)
for col in df.columns:

    pct_missing = np.mean(df[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
data = df[df.columns[0:88]]

data.head()
data.describe()
data.info
print(data.shape)

from sklearn.feature_selection import VarianceThreshold

transform = VarianceThreshold(threshold=0.5)

td = transform.fit(data)

r_list = []

for i, variance in enumerate(td.variances_):

    if variance < 0.5:

        r_list.append(i)

r_list = ["col_" + str(i) for i in r_list]

print(r_list)

selected_data = pd.DataFrame(transform.fit_transform(data))

selected_data['target'] = df['target']

print(selected_data.shape)

selected_data.head()
X = np.array(selected_data.iloc[:, selected_data.columns != 'target'])

y = np.array(selected_data.iloc[:, selected_data.columns == 'target'])
from imblearn.over_sampling import SMOTE



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=120)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
print(len(X_train), len(X_test))
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum( y_train==0)))



sm = SMOTE(random_state=7)

X_train_res, y_train_res = sm.fit_sample(scaled_X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

regressor = LogisticRegression(class_weight="balanced")

regressor.fit(X_train_res, y_train_res)

y_pred = regressor.predict(scaled_X_test)

score = roc_auc_score(y_test, y_pred)

acc  = regressor.score(scaled_X_test, y_test)

print(f"Score : {score} | Accuracy : {acc}")

test_df = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

test_data = test_df.drop('id', axis=1, inplace=False)

for column in r_list:

    test_data.drop(column, axis=1, inplace=True)

print(test_data.shape)

test_data.head()
X_scaled_testf = scaler.transform(test_data)

y_pred = regressor.predict(X_scaled_testf)

submission = pd.DataFrame({'id':test_df['id'],'target':y_pred})

path = '/kaggle/working/predictions_3.csv'

submission.to_csv(path,index=False)

print('Saved file to: ' + path)
print(y_pred)