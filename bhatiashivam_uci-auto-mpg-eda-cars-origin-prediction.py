import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='white', palette='colorblind')
df = pd.read_csv('../input/uci-auto-mpg-dataset/mpg.csv')
df.sample(2)
df.info()
df.isna().sum().plot(kind='bar')
plt.title('Columns with missing values')
df.corr()['horsepower'].sort_values()
df[df.horsepower.isna()].sort_values(by='model_year')
df.groupby(['origin','model_year','fuel_type']).median().loc['usa', 71, 'diesel']
# from above
df.loc[(df.model_year==71) & (df.name=='ford pinto'), 'horsepower'] = 153
# similarly, for other 5 values
df.loc[(df.model_year==74) & (df.name=='ford maverick'), 'horsepower'] = 100
df.loc[(df.model_year==80) & (df.name=='renault lecar deluxe'), 'horsepower'] = 67
df.loc[(df.model_year==80) & (df.name=='ford mustang cobra'), 'horsepower'] = 90
df.loc[(df.model_year==81) & (df.name=='renault 18i'), 'horsepower'] = 81
df.loc[(df.model_year==82) & (df.name=='amc concord dl'), 'horsepower'] = 85.5
# numerical subset
dfnum=df.select_dtypes(include=np.number)
dfnum.drop(columns=['cylinders', 'model_year'], inplace=True) # these two will be later treated as categorical features

# categorical subset
dfcat = pd.concat([df.select_dtypes(exclude=np.number), df[['cylinders', 'model_year']]], axis=1)
dfnum.sample(2)
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 4), sharey=False)
i = 0

for col in axs:
    col.title.set_text(dfnum.columns[i])
    sns.boxplot(data=dfnum.iloc[:,i], ax=col)
    i+=1
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
dfnum_scaled = pd.DataFrame(rs.fit_transform(dfnum), columns=dfnum.columns)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 4))

for col in dfnum_scaled.columns:
    sns.kdeplot(dfnum_scaled[col], ax=ax2)
    
for col in dfnum.columns:
    sns.kdeplot(dfnum[col], ax=ax1)

ax2.title.set_text('After Standardization')
ax1.title.set_text('Before Standardization')
del dfcat['origin'] # deleting target feature
dfcat.sample(2)
dfcat.info()
dfcat['cylinder_cat'] = pd.Categorical(dfcat.cylinders.values, categories=list(dfcat.cylinders.unique()), ordered=False)
dfcat['model_year_cat'] = pd.Categorical(dfcat.model_year.values, categories=list(dfcat.model_year.unique()), ordered=True)
del dfcat['cylinders']
del dfcat['model_year']
del dfcat['name']
dfcat.info()
dfcat_dummy = pd.get_dummies(dfcat, drop_first=True)
dfcat_dummy.sample(2)
df_total = pd.concat([dfnum_scaled, dfcat_dummy], axis=1)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
kfold = StratifiedKFold(n_splits=8)
X, y = df_total, df.origin.map({'usa':1, 'japan':0, 'europe':0})
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
import xgboost as xgb
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

xg_cl.fit(X_train,y_train) 
preds = xg_cl.predict(X_test)

accuracy = float(np.sum(preds==y_test))/y_test.shape[0] 
print("accuracy: %f" % (accuracy)) 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score 

RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [3,"sqrt", "log2"],
              "min_samples_split": [n for n in range(1, 9)],
              "min_samples_leaf": [5, 7],
              "bootstrap": [False, True],
              "n_estimators" :[200, 500],
              "criterion": ["gini", "entropy"]}

gs_rf = GridSearchCV(RFC, param_grid = rf_param_grid, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)

gs_rf.fit(X_train, y_train)

rf_best = gs_rf.best_estimator_
rf_best.fit(X_train, y_train)
y_pred = rf_best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
y_pred_proba = rf_best.predict_proba(X_test)[:,1] 
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc)) 