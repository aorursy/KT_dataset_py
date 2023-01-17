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
autism_data = pd.read_csv('../input/Toddler Autism dataset July 2018.csv', index_col='Case_No')

autism_data.head()
autism_data.info()
autism_data.describe()
autism_data['Sex'][autism_data['Class/ASD Traits '] == 'Yes'].value_counts(normalize=True)*100
# uncomment a line below if you need percentages

# pd.crosstab(autism_data['Ethnicity'], autism_data['Class/ASD Traits ']).apply(lambda r: r/r.sum()*100, axis=1)

pd.crosstab(autism_data['Ethnicity'], autism_data['Class/ASD Traits '])
autism_data['Who completed the test'].unique()
autism_data['Who completed the test'].value_counts()
autism_data['Jaundice'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

sns.countplot(x='Jaundice', hue='Class/ASD Traits ', data=autism_data)
plt.figure(figsize=(12,6))

sns.lmplot('Age_Mons', 'Qchat-10-Score', data=autism_data, hue='Class/ASD Traits ', fit_reg=True)
plt.figure(figsize=(12,6))

sns.countplot(x='Ethnicity', hue='Class/ASD Traits ', data=autism_data)
plt.figure(figsize=(12,6))

sns.countplot(x='Family_mem_with_ASD', hue='Class/ASD Traits ', data=autism_data)
autism_data['Sex'] = autism_data['Sex'].map({'m': 0, 'f': 1})

autism_data['Jaundice'] = autism_data['Jaundice'].map({'no': 0, 'yes': 1})

autism_data['Family_mem_with_ASD'] = autism_data['Family_mem_with_ASD'].map({'no': 0, 'yes': 1})

autism_data['Class/ASD Traits '] = autism_data['Class/ASD Traits '].map({'No': 0, 'Yes': 1})
autism_data['Who completed the test'].replace('Health care professional', 'Health Care Professional', inplace=True);
autism_data.head()
object_cols = ['Ethnicity', 'Who completed the test']



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



label_autism_data = autism_data.copy()



for col in object_cols:

    label_autism_data[col] = label_encoder.fit_transform(autism_data[col])
label_autism_data.head()
X = label_autism_data.drop(['Class/ASD Traits '], axis=1)

y = label_autism_data['Class/ASD Traits ']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold



params = {'max_depth': np.arange(1, 11), 'n_estimators': np.arange(100, 1100, 100)}
forest = RandomForestClassifier(random_state=0)



skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=0)



best_forest = GridSearchCV(estimator=forest, param_grid=params, cv=skf, n_jobs=-1, verbose=1)
best_forest.fit(X_train, y_train)

best_forest.best_params_
best_forest.best_estimator_
best_forest.best_score_
from sklearn.metrics import accuracy_score



accuracy_score(y_test, best_forest.predict(X_test))