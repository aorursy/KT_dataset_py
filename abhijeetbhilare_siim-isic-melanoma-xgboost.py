import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.metrics import accuracy_score

import missingno as msno

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

train.head()
sns.pairplot(train[["age_approx", "target"]])
fig = px.line(train, x="target", y="diagnosis")

py.iplot(fig, filename="simple_line")
ax = sns.countplot(x="age_approx", hue="target", data=train, palette="plasma")

ax.set_title('', fontsize=20)
ax = sns.countplot(x="anatom_site_general_challenge", hue="benign_malignant", data=train, palette="plasma")

ax.set_title('', fontsize=20)
train['sex'] = train['sex'].fillna('na')

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)



test['sex'] = test['sex'].fillna('na')

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)



train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1



test['sex'] = test['sex'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1



train.head()
test.head()
sns.pairplot(test[["sex", "age_approx", "anatom_site_general_challenge"]])
x_train = train[['sex', 'age_approx','anatom_site_general_challenge']]

y_train = train['target']



x_test = test[['sex', 'age_approx','anatom_site_general_challenge']]



train_DMatrix = xgb.DMatrix(x_train, label= y_train)

test_DMatrix = xgb.DMatrix(x_test)
clf = xgb.XGBClassifier(n_estimators=3000, 

                        max_depth=18, 

                        learning_rate=0.15, 

                        num_class = 2, 

                        objective='multi:softprob',

                        seed=0,  

                        nthread=-1, 

                        scale_pos_weight = (32542./584.))



clf.fit(x_train, y_train)
clf.predict_proba(x_test)[:,1]



sub_xgb = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

sub_xgb.target = clf.predict_proba(x_test)[:,1]*1.12
sub_xgb.head()
print(sub_xgb.target.min())

print(sub_xgb.target.max())
sub_xgb.target.hist()
sub_xgb.to_csv('siim_submission.csv', index = False)