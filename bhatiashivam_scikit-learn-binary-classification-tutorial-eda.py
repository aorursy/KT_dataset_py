import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context='notebook', style='white', palette='colorblind')



import os

from pathlib import Path
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
X, y = df_total, df.origin.map({'usa':1, 'japan':0, 'europe':0})
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

import xgboost as xgb
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=0)

gb_cl = GradientBoostingClassifier(random_state=0)
classifiers = [('LogisticRegression', LogisticRegression()), 

               ('RandomForestClassifier', RandomForestClassifier()), 

               ('LinearSVC', LinearSVC()), ('SVC', SVC()), 

               ('KNeighborsClassifier', KNeighborsClassifier()), 

               ('XGBoost', xg_cl), ('GradientBoost', gb_cl)]



#vc = VotingClassifier(estimators=classifiers)

#classifiers.append(('VotingClassifier', VotingClassifier(estimators=classifiers)))
from sklearn.metrics import accuracy_score



dict={}



for clf in classifiers:

    clf[1].fit(X_train, y_train)

    dict[clf[0]] = accuracy_score(y_test, clf[1].predict(X_test))



df_accuracy = pd.DataFrame(dict, index=['Accuracy']).T.sort_values(by='Accuracy', ascending=False)

print(df_accuracy.reset_index())



df_accuracy.plot(kind='bar')

plt.yticks(df_accuracy.values, rotation=0)

plt.ylim(0.77, 0.93)

plt.plot()
importances = pd.Series(data=clf[1].feature_importances_, index= X_train.columns) 



importances_sorted = importances.sort_values() 

importances_sorted.plot(kind='barh', color='lightgreen') 



plt.title('Random Forest: Features Importances') 

plt.show() 