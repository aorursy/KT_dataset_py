import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import make_scorer

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df2 = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df_test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df_test2 = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df.head()

num_rows = df.shape[0]

num_cols = df.shape[1]

(num_rows, num_cols)
df.info()
df.describe()         

# numerical variables
df.isnull().any()
missing_values = df.isnull().sum()

missing_values[missing_values > 0]
sel_features = ['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']

x1 = df[sel_features].copy()

y1 = df['class'].copy()

x1_t = df_test[sel_features].copy()



sel_features_2 = ['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']

x2 = df2[sel_features_2].copy()

y2 = df2['class'].copy()

x2_t = df_test[sel_features_2].copy()
scaler = RobustScaler()

x1_scaled = scaler.fit_transform(x1[sel_features])

x1_t_scaled = scaler.transform(x1_t[sel_features])
est1 = [('rf', RandomForestClassifier()), ('bag', DecisionTreeClassifier()), ('xgb', XGBClassifier())]



hard_voter = VotingClassifier(estimators=est1, voting='hard').fit(x1,y1)

y1_pred1 = hard_voter.predict(x1_t);
for i in range(len(y1_pred1)):

    y1_pred1[i] = round(y1_pred1[i])

id_new = df_test['id'].copy()

output_new = pd.DataFrame(list(zip(id_new,y1_pred1)), columns = ['id','class'])
con_dic1 = {'id': int, 'class': int} 
output_new = output_new.astype(con_dic1) 



output_new.to_csv('res_g.csv', index = False)
scaler = RobustScaler()

x2_scaled = scaler.fit_transform(x2[sel_features_2])

x2_scaled_t = scaler.transform(x2_t[sel_features_2])
xgC2 = XGBClassifier()

xgC2.fit(x2_scaled,y2.values.ravel())
y_pred2_x2 = xgC2.predict(x2_scaled_t)
for i in range(len(y_pred2_x2)):

    y_pred2_x2[i] = round(y_pred2_x2[i])
x2_id_n = df_test2['id'].copy()

x2_output_n = pd.DataFrame(list(zip(x2_id_n,y_pred2_x2)), columns = ['id','class'])
con_dic2 = {'id': int, 'class': int} 
x2_output_n = x2_output_n.astype(con_dic2) 