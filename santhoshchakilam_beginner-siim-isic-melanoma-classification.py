import numpy as np
import pandas as pd
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train
train.info()
train.isnull().sum()
train['sex'] = train['sex'].fillna('unknown')
train['age_approx'] = train['age_approx'].fillna(train['age_approx'].median())
train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('unknown')
train.isnull().sum()
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
test
test.isnull().sum()
test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('unknown')
test.isnull().sum()
train.drop(['diagnosis', 'benign_malignant'],axis=True,inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['image_name'] = le.fit_transform(train['image_name'])
train['patient_id'] = le.fit_transform(train['patient_id'])
train['sex'] = le.fit_transform(train['sex'])
train['anatom_site_general_challenge'] = le.fit_transform(train['anatom_site_general_challenge'])
train
X = train.drop(['target'], axis=1)
y = train.target
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier

model = LGBMClassifier()


model.fit(X,y)
z = test
z
z['image_name'] = le.fit_transform(z['image_name'])
z['patient_id'] = le.fit_transform(z['patient_id'])
z['sex'] = le.fit_transform(z['sex'])
z['anatom_site_general_challenge'] = le.fit_transform(z['anatom_site_general_challenge'])
z

z
target = model.predict(z)
x = pd.DataFrame(target,columns=['target'])
q = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
y = q.join(x)
y
y = y.drop(['patient_id','sex','age_approx','anatom_site_general_challenge'],axis=1)
y
y.to_csv('siim.csv', index=False)
