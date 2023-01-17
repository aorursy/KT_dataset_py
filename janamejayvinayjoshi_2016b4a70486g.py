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

import xgboost

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
df_train_path = '/kaggle/input/eval-lab-2-f464/train.csv'

df_test_path = '/kaggle/input/eval-lab-2-f464/test.csv'

df_tr = pd.read_csv(df_train_path)

df_te = pd.read_csv(df_test_path)
sns.countplot(x='class',data=df_tr)
sns.boxplot(x = 'class',y ='chem_0',data = df_tr)
sns.boxplot(x = 'class',y ='chem_1',data = df_tr)
sns.boxplot(x = 'class',y ='chem_2',data = df_tr)
sns.boxplot(x = 'class',y ='chem_3',data = df_tr)
sns.boxplot(x = 'class',y ='chem_4',data = df_tr)
sns.boxplot(x = 'class',y ='chem_5',data = df_tr)
sns.boxplot(x = 'class',y ='chem_6',data = df_tr)
sns.boxplot(x = 'class',y ='chem_7',data = df_tr)
sns.boxplot(x = 'class',y ='attribute',data = df_tr)
df = df_tr[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute","class"]].copy()

# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
### FEATURE SELECTION FOR VOTING CLASSIFIER

feat_vc = ['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']

X_vc = df_tr[feat_vc].copy()

Y_vc = df_tr['class'].copy()

X_vc_t = df_te[feat_vc].copy()



### FEATURE SCALING FOR VOTING CLASSIFIER

scaler = RobustScaler()

X_vc_sc = scaler.fit_transform(X_vc[feat_vc])

X_vc_t_sc = scaler.transform(X_vc_t[feat_vc])



### SELECTING COMPETING CLASSIFIERS FOR VOTING CLASSIFIER

estimators = [('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('xgb', XGBClassifier())]



### FITTING AND PREDICTION

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_vc_sc,Y_vc)

Y_vc_pred = hard_voter.predict(X_vc_t_sc);



### PREPARING OUTPUT DATAFRAME

for i in range(len(Y_vc_pred)):

    Y_vc_pred[i] = round(Y_vc_pred[i])

vc_new_id = df_te['id'].copy()

vc_output = pd.DataFrame(list(zip(vc_new_id,Y_vc_pred)), columns = ['id','class'])

convert_dict_vc = {'id': int, 'class': int}  

vc_output = vc_output.astype(convert_dict_vc) 



### OUTPUT DATAFRAME AS CSV

vc_output.to_csv('VoteC.csv', index = False)
### FEATURE SELECTION FOR XGBoost CLASSIFIER

feat_xg = ['chem_1','chem_2','chem_4','chem_5','chem_6','attribute']

X_xg = df_tr[feat_xg].copy()

Y_xg = df_tr['class'].copy()

X_t_xg = df_te[feat_xg].copy()



### FEATURE SCALING FOR XGBoost CLASSIFIER

scaler = RobustScaler()

X_sc_xg = scaler.fit_transform(X_xg[feat_xg])

X_t_sc_xg = scaler.transform(X_t_xg[feat_xg])



### FITTING AND PREDICTION

xg = XGBClassifier()

xg.fit(X_sc_xg,Y_xg.values.ravel())

Y_pred_xg = xg.predict(X_t_sc_xg)



### PREPARING OUTPUT DATAFRAME

for i in range(len(Y_pred_xg)):

    Y_pred_xg[i] = round(Y_pred_xg[i])

xg_new_id = df_te['id'].copy()

xg_output = pd.DataFrame(list(zip(xg_new_id,Y_pred_xg)), columns = ['id','class'])

convert_dict_xg = {'id': int, 'class': int}   

xg_output = xg_output.astype(convert_dict_xg) 



### OUTPUT DATAFRAME AS CSV

xg_output.to_csv('XG_feat_3_def.csv', index = False)