import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/thoraric-surgery/ThoraricSurgery.csv', index_col = 'id')
data.head(2)
data.info()
data[['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']] = \
(data[['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']] == 'T').astype(int)
data.head(2)
data['DGN']   = data['DGN'].str[-1:].astype(int)
data['PRE6']  = data['PRE6'].str[-1:].astype(int)
data['PRE14'] = data['PRE14'].str[-1:].astype(int)
data.describe(include='all')
col = ['Diagnosis','Forced_Capacity','Forced_Expiration','Zubrod_scale','Pain',' Haemoptysis','Dyspnoea',
       'Cough','Weakness','Size_of_tumor','diabetes','MI_6months','PAD','Smoker','Asthmatic','Age','Risk_1y']
data.columns = col
data.head()
sns.pairplot(data[['Forced_Expiration','Smoker', 'Age', 'Risk_1y']], 
             hue='Risk_1y', diag_kws={'bw':1.5}, markers=['o', 'D'], height=2.5)
fig, ax = plt.subplots(figsize=(12, 12))
mask=np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data.corr(), annot=True, linewidths=.1, cmap="YlGnBu", square=True, mask=mask, cbar=False)
fig, ax = plt.subplots(figsize = (12,6))
sns.scatterplot(x='Diagnosis', y='Size_of_tumor', #hue='Risk_1y', 
                size='dgn_cnt', sizes=(20, 250),
                data=data.groupby(['Diagnosis','Size_of_tumor']).size().reset_index().rename(columns={0:'dgn_cnt'}))
fig, ax = plt.subplots(figsize = (10,6))
sns.barplot(x='Diagnosis', y='Risk_1y', 
            data = data, palette="Blues_d",
            ax=ax, ci=None)
# но и наблюдений таких совсем мало..
data[data.Diagnosis.isin([1, 6])]
fig, ax = plt.subplots(figsize = (10,6))
sns.distplot(data.Diagnosis, kde=False)
data1 = data.copy()
data1['Diagnosis'] = np.where(data['Diagnosis'].isin([1,5,6,7,8]), 0, data['Diagnosis'])
data1.head()
fig, ax = plt.subplots(figsize = (10,6))
sns.distplot(data1.Diagnosis, kde=False)
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
X = data1.drop(columns='Diagnosis')
y = data1.Diagnosis
clf = LogisticRegression(class_weight = 'balanced')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(metrics.classification_report(y_test, predictions))
from imblearn.over_sampling import SMOTE, ADASYN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
augm = ADASYN()
X_train_augm, y_train_augm = augm.fit_resample(np.array(X_train), np.array(y_train))
pd.Series(y_train).value_counts()
pd.Series(y_train_augm).value_counts()
clf = LogisticRegression()
clf.fit(X_train_augm, y_train_augm)
predictions = clf.predict(X_test)
print(metrics.classification_report(y_test, predictions))