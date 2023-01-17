# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tr=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv",header=None)

ts=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv",header=None)

trlabel=pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv",header=None,names=["Target"])
tr.head()
tr.shape
trlabel.shape
tr=pd.concat([tr,trlabel],axis=1) ## Concatenation along columns
tr.info() ## No null values
fig = plt.figure(figsize=(20, 15))

for j,i in enumerate(tr.columns[:-1]):

    fig.add_subplot(8, 5, j+1)

    sns.distplot(tr[i])
tr.Target.value_counts(normalize=True).plot(kind='bar'); ## Not an imbalanced data
fig = plt.figure(figsize=(20, 15))

for j,i in enumerate(tr.columns[:-1]):

    fig.add_subplot(8, 5, j+1)

    sns.boxplot(x=tr[i]) ## looking at the boxplots,we can see that there are outliers
for i in tr.columns[:-1]:

    Q1=tr[i].quantile(0.25)

    Q3=tr[i].quantile(0.75)

    IQR=Q3-Q1

    IQR

    tr.loc[(tr[i]<(Q1-1.5*IQR))|(tr[i]>(Q3+(1.5*IQR))),i] = np.nan

    tr[i].fillna(tr[i].mean(),inplace=True)
fig = plt.figure(figsize=(20, 15))

for j,i in enumerate(tr.columns[:-1]):

    fig.add_subplot(8, 5, j+1)

    sns.boxplot(x=tr[i]) ## looking at the boxplots,we can see that ouliers are removed
from statsmodels.stats.outliers_influence import variance_inflation_factor



X = tr.loc[:, tr.columns != 'Target']

X["intecept"] = 1



vif = pd.DataFrame()

vif["variables"] = X.columns

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



print(vif)
tr.drop(columns=[4,12,23],inplace=True) ## removing the columns with concerning multicollinearity
print(tr.head())

print(tr.shape) ## Total of 38 columns and 1000 rows
X = tr.loc[:, tr.columns != 'Target']

X["intecept"] = 1



vif = pd.DataFrame()

vif["variables"] = X.columns

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



print(vif) ## here all the VIF values are around the value 1, Therefore we are good to go further
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix, f1_score

from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
X_train, X_val, y_train, y_val=train_test_split(tr.iloc[:,:-1], tr.iloc[:,-1], stratify=tr["Target"], random_state=2000)
print('X_train:', X_train.shape )

X_train.head()
print('X_val:', X_val.shape )

X_val.head()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(X_train, y_train)

y_train_pred_lr=lr.predict(X_train)

y_val_pred_lr=lr.predict(X_val)
print('\n*** Cross Val Accuracy ***\n')

print('Log Reg',cross_val_score(lr, tr.iloc[:,:-1], tr.iloc[:,-1], cv=10).mean())



print('\n*** Train Test Metrics ***\n')

print('Train Accuracy score:', accuracy_score(y_train,y_train_pred_lr))

print('Val Accuracy score:', accuracy_score(y_val,y_val_pred_lr))

print('Train Precision score:', precision_score(y_train,y_train_pred_lr))

print('Val Precision score:', precision_score(y_val,y_val_pred_lr))

print('Train Recall score:', recall_score(y_train,y_train_pred_lr))

print('Val Recall score:', recall_score(y_val,y_val_pred_lr))

print('Train f1 score:', f1_score(y_train,y_train_pred_lr))

print('Val f1 score:', f1_score(y_val,y_val_pred_lr))

print('Train roc auc score:', roc_auc_score(y_train,y_train_pred_lr))

print('Val roc auc score:', roc_auc_score(y_val,y_val_pred_lr))

print('Train confusion matrix:\n', confusion_matrix(y_train,y_train_pred_lr))

print('Val confusion matrix:\n', confusion_matrix(y_val,y_val_pred_lr))
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=4)

dt.fit(X_train, y_train)

y_train_pred_dt=dt.predict(X_train)

y_val_pred_dt=dt.predict(X_val)
print('\n*** Cross Val Accuracy ***\n')

print('Dec Tree',cross_val_score(dt, tr.iloc[:,:-1], tr.iloc[:,-1], cv=10).mean())



print('\n*** Train Test Metrics ***\n')

print('Train Accuracy score:', accuracy_score(y_train,y_train_pred_dt))

print('Val Accuracy score:', accuracy_score(y_val,y_val_pred_dt))

print('Train Precision score:', precision_score(y_train,y_train_pred_dt))

print('Val Precision score:', precision_score(y_val,y_val_pred_dt))

print('Train Recall score:', recall_score(y_train,y_train_pred_dt))

print('Val Recall score:', recall_score(y_val,y_val_pred_dt))

print('Train f1 score:', f1_score(y_train,y_train_pred_dt))

print('Val f1 score:', f1_score(y_val,y_val_pred_dt))

print('Train roc auc score:', roc_auc_score(y_train,y_train_pred_dt))

print('Val roc auc score:', roc_auc_score(y_val,y_val_pred_dt))

print('Train confusion matrix:\n', confusion_matrix(y_train,y_train_pred_dt))

print('Val confusion matrix:\n', confusion_matrix(y_val,y_val_pred_dt))
params = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],

          'max_depth':[1,2,3,4,5,6,7,8,9],

         'criterion':['gini','entropy'],}

#Making models with hyper parameters sets

model1 = GridSearchCV(dt, param_grid=params, n_jobs=-1, cv = 10)

#Learning

model1.fit(X_train, y_train)

#The best hyper parameters set

print("Best Hyper Parameters:",model1.best_params_)



#Prediction

y_val_pred_dt_tuned = model1.predict(X_val)

print("Accuracy:",accuracy_score(y_val_pred_dt_tuned,y_val))
from sklearn.ensemble import RandomForestClassifier
rfr=RandomForestClassifier(max_depth=4)

rfr.fit(X_train, y_train)

y_train_pred_rfr=rfr.predict(X_train)

y_val_pred_rfr=rfr.predict(X_val)
print('\n*** Cross Val Accuracy ***\n')

print('RFR',cross_val_score(rfr, tr.iloc[:,:-1], tr.iloc[:,-1], cv=10).mean())



print('\n*** Train Test Metrics ***\n')

print('Train Accuracy score:', accuracy_score(y_train,y_train_pred_rfr))

print('Val Accuracy score:', accuracy_score(y_val,y_val_pred_rfr))

print('Train Precision score:', precision_score(y_train,y_train_pred_rfr))

print('Val Precision score:', precision_score(y_val,y_val_pred_rfr))

print('Train Recall score:', recall_score(y_train,y_train_pred_rfr))

print('Val Recall score:', recall_score(y_val,y_val_pred_rfr))

print('Train f1 score:', f1_score(y_train,y_train_pred_rfr))

print('Val f1 score:', f1_score(y_val,y_val_pred_rfr))

print('Train roc auc score:', roc_auc_score(y_train,y_train_pred_rfr))

print('Val roc auc score:', roc_auc_score(y_val,y_val_pred_rfr))

print('Train confusion matrix:\n', confusion_matrix(y_train,y_train_pred_rfr))

print('Val confusion matrix:\n', confusion_matrix(y_val,y_val_pred_rfr))
params = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4], 

          'min_samples_leaf':[2,3,4,],

          'max_depth':[1,2,3,4,5,6,7,8,9],

         'criterion':['gini','entropy'], 

         'n_estimators' :[150]}

#Making models with hyper parameters sets

model2 = GridSearchCV(rfr, param_grid=params, n_jobs=-1, cv = 10)

#Learning

model2.fit(X_train, y_train)

#The best hyper parameters set

print("Best Hyper Parameters:",model2.best_params_)



#Prediction

y_val_pred_rfr_tuned = model2.predict(X_val)

print("Accuracy:",accuracy_score(y_val_pred_rfr_tuned,y_val))
from sklearn.neighbors import KNeighborsClassifier
## Trying to find the best no of neighbours.

cross_val_accuracy_per_neighbours = []

for i in range(1, 30):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(tr.iloc[:,:-1], tr.iloc[:,-1])

    cross_val_accuracy = np.mean(cross_val_score(knn, tr.iloc[:,:-1], tr.iloc[:,-1], cv=10))

    cross_val_accuracy_per_neighbours.append([i, cross_val_accuracy])
max_df = pd.DataFrame(cross_val_accuracy_per_neighbours, columns = ['Neighbour', 'Mean Accuracy']).sort_values(by='Mean Accuracy', ascending=False)

max_df.head() ## Here we get the maximum mean accuracy for 10 neighbours.
knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(X_train,y_train)

y_train_pred_knn = knn.predict(X_train)

y_val_pred_knn = knn.predict(X_val)
print('\n*** Cross Val Accuracy ***\n')

print('KNN',cross_val_score(knn, tr.iloc[:,:-1], tr.iloc[:,-1], cv=10).mean(),"(**Best Accuracy yet**)")



print('\n*** Train Test Metrics ***\n')

print('Train Accuracy score:', accuracy_score(y_train,y_train_pred_knn))

print('Val Accuracy score:', accuracy_score(y_val,y_val_pred_knn))

print('Train Precision score:', precision_score(y_train,y_train_pred_knn))

print('Val Precision score:', precision_score(y_val,y_val_pred_knn))

print('Train Recall score:', recall_score(y_train,y_train_pred_knn))

print('Val Recall score:', recall_score(y_val,y_val_pred_knn))

print('Train f1 score:', f1_score(y_train,y_train_pred_knn))

print('Val f1 score:', f1_score(y_val,y_val_pred_knn))

print('Train roc auc score:', roc_auc_score(y_train,y_train_pred_knn))

print('Val roc auc score:', roc_auc_score(y_val,y_val_pred_knn))

print('Train confusion matrix:\n', confusion_matrix(y_train,y_train_pred_knn))

print('Val confusion matrix:\n', confusion_matrix(y_val,y_val_pred_knn))
lift_df=tr.copy()

X=lift_df.iloc[:,:-1]

Y=lift_df.iloc[:,-1]

lift_df["Probability"]=knn.predict_proba(X)[:,1]

lift_df.sort_values(by="Probability",ascending=False,inplace=True)

lift_df['Deciles']=pd.cut(lift_df["Probability"],10)

ldf=pd.crosstab(lift_df["Deciles"],lift_df["Target"]).sort_values(by=1,ascending=False)

ldf["Cummalative Ones"]=ldf[1].cumsum().values

ldf["% of Ones"]=(ldf[1]*100/sum(ldf[1]))

ldf["Gain"]=ldf["% of Ones"].cumsum().values

ldf['Non Model Gain'] = np.arange(10, 101, 10)

#size of each decile

size_decile=lift_df.groupby("Deciles")["Target"].count().values

ldf["Cummalative Lift"]=ldf["Gain"]/size_decile

ldf.index=['1','2','3','4','5','6','7','8','9','10']
ldf
plt.plot(ldf.index, ldf['Gain'].values, color='darkorange')

plt.plot(ldf.index, ldf['Non Model Gain'].values, color='black');