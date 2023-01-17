# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/minor-project-2020/train.csv",header=None, delimiter=",")

df.columns=df.iloc[0]

df=df.drop(df.index[0])

df.head()
columns_to_drop=['id']

df=df.drop(columns=columns_to_drop)
df.head()
df_X_train = df.iloc[:, 0:-1].astype(float)

df_y_train = df.iloc[:,-1].astype(int)
df_X_train.head()
df_y_train.head()
matrix = np.triu(df_X_train.corr())

fig_dims = (60, 36)

#double click on the heatmap to enlarge

fig, ax = plt.subplots(figsize=fig_dims)

sns.heatmap(df_X_train.corr(),annot=True,vmin=-1, vmax=1, center= 0,mask=matrix)
#selecting based on correlation (correlation greater than 0.95)

corr = df_X_train.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.95:

            if columns[j]:

                columns[j] = False

selected_columns = df_X_train.columns[columns]

df_X = df_X_train[selected_columns]

df_X.head()
X_train = df_X

y_train = df_y_train
from collections import Counter

counter = Counter(y_train)

print(counter)
weights = {0:1,1:530}
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold



model = LogisticRegression(solver='lbfgs',verbose=1,class_weight=weights,max_iter=10000)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores))
model.fit( X_train, y_train)
'''from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()

X_train = sc_X.fit_transform(X_train)'''

'''from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=69)

X_train, y_train = sm.fit_resample(X_train, y_train)'''

'''from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state = 0,verbose=1)

model.fit(X_train,y_train)'''
'''from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(loss='exponential',n_estimators = 100, random_state = 0,verbose=1)

model.fit(X_train,y_train)'''
'''from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,y_train)'''
df_test = pd.read_csv("../input/minor-project-2020/test.csv",header=None, delimiter=",")

df_test.columns=df_test.iloc[0]

df_test=df_test.drop(df_test.index[0])

df_test=df_test.reset_index()

id_val = df_test['id']

df_test = df_test[selected_columns]

df_test.head()
X_test = df_test

#X_test = sc_X.transform(X_test)
y_pred = model.predict_proba(X_test)

#y_pred[y_pred < 0] = 0

#y_pred = np.around(y_pred)

#y_pred = y_pred.astype(int)

y_pred = y_pred[:,1]

y_pred = pd.Series(y_pred)

y_pred
'''counter = Counter(y_pred)

print(counter)'''
frame = { 'id': id_val, 'target': y_pred } 

df_final = pd.DataFrame(frame)

df_final.to_csv("submission.csv",index=False)

df_final