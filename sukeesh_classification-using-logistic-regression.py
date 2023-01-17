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
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df.head()
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

df.diagnosis.unique()
for col_name in df:

    if df[col_name].isnull().values.any() == True:

        print(col_name, " => " ,(df[col_name].isnull().sum().sum()/df.shape[0])*100)

        if (df[col_name].isnull().sum().sum()/df.shape[0])*100 == 100.0:

            df=df.drop([col_name], axis=1)

df.head(1)

df=df.drop(['id'], axis=1)
# for col_name in df:

#     if col_name != 'diagnosis':

#         df.plot.scatter(x=col_name, y='diagnosis')
def normalize(df):

    result = df.copy()

    for col_name in df:

        max_value = df[col_name].max()

        min_value = df[col_name].min()

        result[col_name] = (df[col_name] - min_value) / (max_value - min_value)

    return result



df = normalize(df)

df.head(5)
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



model = LogisticRegression()

scaler = StandardScaler()

lr = LogisticRegression()

model1 = Pipeline([('standardize', scaler),

                    ('log_reg', lr)])
from sklearn.model_selection import train_test_split

features = []

for col_name in df:

    if col_name != 'diagnosis':

        features.append(col_name)

x = df[features]

y = df['diagnosis']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=3)
model1.fit(x_train, y_train)
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 

y_test_hat = model1.predict(x_test)

y_test_hat_probs = model1.predict_proba(x_test)[:,1]

test_accuracy = accuracy_score(y_test, y_test_hat)*100

test_auc_roc = roc_auc_score(y_test, y_test_hat_probs)*100

print('Confusion matrix:\n', confusion_matrix(y_test, y_test_hat))

print('Training accuracy: %.4f %%' % test_accuracy)

print('Training AUC: %.4f %%' % test_auc_roc)