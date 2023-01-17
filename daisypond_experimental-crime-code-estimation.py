import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

import sklearn as sk

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/crime.csv', encoding='ISO-8859-1')

df.head()
df.info()
df = df[df['UCR_PART'] == 'Part One']

df.nunique()
df2 =df.loc[:, ['OFFENSE_CODE_GROUP', 'DISTRICT', 'REPORTING_AREA', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'STREET','Lat','Long']]
s = df2['OFFENSE_CODE_GROUP'].value_counts(normalize=True)

s2 = pd.DataFrame(s)

s2.plot.pie(y='OFFENSE_CODE_GROUP',figsize=(5, 5), legend=False, counterclock=False, startangle=0, autopct="%1.1f%%")

plt.show()
str =['OFFENSE_CODE_GROUP', 'DISTRICT', 'REPORTING_AREA', 'DAY_OF_WEEK', 'STREET']



for column in str:

    labels, uniques = pd.factorize(df2[column])

    print(uniques)

    df2[column] = labels

df2.nunique()
df2.loc[df2['OFFENSE_CODE_GROUP'] == 3, 'OFFENSE_CODE_GROUP'] = 0

df2.loc[df2['OFFENSE_CODE_GROUP'] > 0, 'OFFENSE_CODE_GROUP'] = 1

df2.head()
df2 = df2.drop('YEAR', axis=1)

df_train, df_test = train_test_split(df2, test_size=0.1)

X_train = df_train.iloc[:,1:] 

y_train = df_train.iloc[:,:1]

X_test = df_test.iloc[:,1:] 

y_test = df_test.iloc[:,:1].values.flatten()
param_grid = {

'learning_rate':[0.1],

'n_estimators':[50,100],

'max_depth':[5,10],

'min_child_weight':[1,2,3],

'max_delta_step':[5],

'gamma':[0.001, 0.01, 0.1,1,10],

'subsample':[0.8],

'colsample_bytree':[0.8],

'objective':['binary:logistic'],

'nthread':[4],

'scale_pos_weight':[1],

'seed':[0],

'scoring':['roc_auc'],

'tree_method':['gpu_hist']

}



grid_search = GridSearchCV(xgb.XGBClassifier(tree_method='gpu_hist',verbosity=2),param_grid, cv=5, verbose=3)

grid_search.fit(X_train, y_train)
grid_search.score(X_test,y_test)
y_test_pred = grid_search.predict(X_test)



cm = confusion_matrix(y_test, y_test_pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



labels={'Larceny', 'others'}



df_cmx = pd.DataFrame(cm)

sns.heatmap(df_cmx, annot=True, xticklabels=labels, yticklabels=labels)

plt.show()