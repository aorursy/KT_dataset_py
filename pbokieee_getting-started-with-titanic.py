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
tr_data = pd.read_csv('/kaggle/input/titanic/train.csv')

te_data = pd.read_csv('/kaggle/input/titanic/test.csv')
tr_data.info()
tr_data.head()
te_data.info()
te_data.head()
tr_data = tr_data.drop(['Cabin','Embarked'],axis = 1)

te_data = te_data.drop(['Cabin','Embarked'],axis = 1)
tr_data = tr_data.drop(tr_data[tr_data.Age.isna()].index)

#te_data = te_data.drop(te_data[te_data.Age.isna()].index)

te_data.Age.fillna(te_data.Age.mean(),inplace = True)

te_data.Fare.fillna(0,inplace = True)
object_cols = [cname for cname in tr_data.columns if tr_data[cname].dtype == "object"]



low_cardinality_cols  = [cname for cname in tr_data.columns if tr_data[cname].nunique() < 10 and 

                        tr_data[cname].dtype == "object"]

numerical_cols = [cname for cname in tr_data.columns if tr_data[cname].dtype in ['int64', 'float64']]



print(object_cols)

print(low_cardinality_cols)

print(numerical_cols)
nouse_object_cols = list(set(object_cols)-set(low_cardinality_cols))

tr_data = tr_data.drop(nouse_object_cols,axis = 1)

te_data = te_data.drop(nouse_object_cols,axis = 1)
from sklearn.preprocessing import LabelEncoder

label_tr_data = tr_data.copy()

label_te_data = te_data.copy()

label_encoder = LabelEncoder()

label_tr_data.Sex = label_encoder.fit_transform(tr_data['Sex'])

label_te_data.Sex = label_encoder.fit_transform(te_data['Sex'])
label_tr_data.head()
label_te_data.head()
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

y = label_tr_data.Survived

X = label_tr_data.drop('Survived',axis = 1)



model = RandomForestClassifier(criterion = 'entropy',n_estimators=100,n_jobs = 16)

model.fit(X, y)



importances = model.feature_importances_





import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

 

features_list = label_tr_data.columns.values

feature_importance = model.feature_importances_

sorted_idx = np.argsort(feature_importance)

 

plt.figure(figsize=(5,7))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()







predictions = model.predict(label_te_data)



output = pd.DataFrame({'PassengerId': label_te_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
