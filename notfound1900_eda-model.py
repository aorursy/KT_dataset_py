import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

import seaborn as sns

import keras 

import matplotlib as mpl
sns.set_style('darkgrid')

path = '../input/creditcardfraud/creditcard.csv'

df = pd.read_csv(path)
df
feature_anomy = df.iloc[:,1:29]

feature_anomy = pd.concat((feature_anomy,df['Class']),axis = 1)

feature_anomy = pd.melt(feature_anomy,id_vars='Class',var_name='Anonyme Feature')

grid = sns.FacetGrid(feature_anomy,col='Anonyme Feature',col_wrap=4,sharex=False,sharey=False,hue = 'Class',)

grid.map(sns.distplot,'value').add_legend()
plt.figure(figsize=(8,6))

plt.hist(df['Amount'],color = 'lightblue',range=(0,500))

plt.show()
#plt.figure(figsize=(8,6))

sns.distplot(df['Time'])
from sklearn.preprocessing import RobustScaler

robu_scaler = RobustScaler()
df.loc[:,['Amount','Time']] = robu_scaler.fit_transform(df.loc[:,['Amount','Time']])
df
sns.choose_colorbrewer_palette('d')
from sklearn.model_selection import train_test_split 

x = np.array(df.iloc[:, df.columns != 'Class'])

y = np.array(df.iloc[:, df.columns == 'Class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report,accuracy_score

rf = RandomForestClassifier(n_estimators=500,n_jobs=-1,max_depth=20)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred)



sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Spectral", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



labels = ['Non-fraud', 'Fraud']

print(classification_report(y_test, y_pred, target_names=labels))
feture_names = [fea for fea in df.columns if fea !='Class']

imporance = rf.feature_importances_

feature_importances_ = pd.DataFrame({'Name':feture_names,'Importances':imporance})
feature_importances_ = feature_importances_.sort_values(by = 'Importances',ascending=False)
plt.figure(figsize=(14,8))

plt.bar(feature_importances_['Name'],feature_importances_['Importances'],color = 'c')
precision, recall, thresholds = precision_recall_curve(y_test,rf.predict_proba(x_test)[:,1])

plt.plot(recall,precision,color = 'c')

plt.xlabel('recall')

plt.ylabel('precision')
accuracy_score(y_pred,y_test)
from imblearn.over_sampling import SMOTE

smote = SMOTE()
sampled_x,sampled_y = smote.fit_resample(x_train,y_train)
sns.countplot(sampled_y)
rf.fit(sampled_x, sampled_y)

y_pred = rf.predict(x_test)
confusion_matrix = confusion_matrix(y_test, y_pred)



sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Spectral", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



labels = ['No-fraud', 'Fraud']

print(classification_report(y_test, y_pred, target_names=labels))
feture_names = [fea for fea in df.columns if fea !='Class']

imporance = rf.feature_importances_

feature_importances_ = pd.DataFrame({'Name':feture_names,'Importances':imporance})

feature_importances_ = feature_importances_.sort_values(by = 'Importances',ascending=False)
plt.figure(figsize=(14,8))

plt.bar(feature_importances_['Name'],feature_importances_['Importances'],color = 'lightblue')
precision, recall, thresholds = precision_recall_curve(y_test,rf.predict_proba(x_test)[:,1])

plt.plot(recall,precision,color = 'c')

plt.xlabel('recall')

plt.ylabel('precision')
accuracy_score(y_pred,y_test)
from lightgbm import LGBMClassifier

import lightgbm as lgb
param = {'boosting_type': 'gbdt',

         'num_leaves': 60,

         'max_depth': -1,

         'min_data_in_leaf': 20,   

         'objective': 'binary',

         'learning_rate': 0.02,

         "min_child_samples": 20,

         'n_estimators':800,

         "feature_fraction": 0.8,

         "bagging_freq": 1,

         "bagging_fraction": 0.8,

         "bagging_seed": 11,

         "metric":'binary_logloss'

         }
sampled_y = sampled_y.reshape(-1,1)
model = lgb.LGBMClassifier(**param)

model.fit(sampled_x,sampled_y,eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=50)
y_pred = model.predict(x_test,num_iteration=model.best_iteration_)
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test,y_pred)


sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="Spectral", fmt='g')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')



labels = ['No-fraud', 'Fraud']

print(classification_report(y_test, y_pred, target_names=labels))
accuracy_score(y_test,y_pred)