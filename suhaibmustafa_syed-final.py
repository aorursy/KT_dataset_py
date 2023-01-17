# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)
df_train.head()
#df_train = df_train[df_train['capital-gain'] < df_train['capital-gain'].quantile(.99)]
#df_train = df_train[df_train['capital-loss'] < df_train['capital-loss'].quantile(.99)]
#df_train = df_train[(df_train['native-country'].str.strip() != "?") & 
#      (df_train['workclass'].str.strip() != "?") & (df_train['occupation'].str.strip() != "?") ]
#df_train.describe()
df_test['target'] = np.nan

df = pd.concat([df_train, df_test])

print(df.shape)
df.head()
df.tail()
#df['native-country']=df['native-country'].apply(lambda x: 'US' if (x == ' United-States')
 #                                               else 'Non-US')
#df.head()
#df['hours-per-week']=df['hours-per-week'].apply(lambda x: 
#                                      'high' if (x > 40) else ('low' if x<30 else 'normal') )
loss_profit_plot = sns.countplot(df['education'], hue=df['target'])
loss_profit_plot.set_xticklabels(loss_profit_plot.get_xticklabels(), rotation=90);
df[['education-num','education']].drop_duplicates()
df['education-level']=df['education-num'].apply(lambda x: 
                                      'HS_Assoc' if (x < 13.0) else 
                                                ('Bach_Mast' if x < 15.0 else 'Doc'))
loss_profit_plot = sns.countplot(df['education-level'], hue=df['target'])
loss_profit_plot.set_xticklabels(loss_profit_plot.get_xticklabels(), rotation=90);
df['age']=df['age'].apply(lambda x: 'young_adult' if (x < 31) else 
                                                ('middle_adult' if x < 60 else 'old_adult'))
loss_profit_plot = sns.countplot(df['age'], hue=df['target'])
loss_profit_plot.set_xticklabels(loss_profit_plot.get_xticklabels(), rotation=90);
#df['marital-status']=df['marital-status'].apply(lambda x: 'Married' if (x.startswith(" Married") & (x != ' Married-spouse-absent'))
#                                                else 'Single')
#diff = df['capital-gain']-df['capital-loss']
#df['loss-profit']=diff.apply(lambda x: 
#                                      'high_profit' if (x > 15000) else 
 #                                        ('profit' if x>0 else 
 #                                        ('breakeven' if x==0 else 
 #                                        ('loss' if x> -2250 else 'high_loss'))))
df.dtypes
df_tmp = df.loc[
    df['target'].notna()
].groupby(
    ['education']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean', 'std': 'target_std'}
).fillna(0.0).reset_index()


df = pd.merge(
    df,
    df_tmp,
    how='left',
    on=['education']
)

df.shape
df['target_mean'] = df['target_mean'].fillna(0.0)
df['target_std'] = df['target_std'].fillna(0.0)
df.drop(columns=['uid','education-num','education','race','workclass','relationship'],inplace=True)
df.columns
df = pd.get_dummies(
    df, 
    columns=[c for c in df.columns if df[c].dtype == 'object'],
    drop_first=True
)
df.head()
len(df.columns)
#Cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'], 
                                                    test_size=0.30, 
                                                    random_state = 100)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 10),
    'min_samples_leaf': range(15, 25, 3),
    'min_samples_split': range(30, 45, 5),
    'criterion': ["entropy", "gini"]
}

LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1, scoring=LogLoss)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results
from sklearn.metrics import log_loss
log_loss(y_test, grid_search.predict_proba(X_test)[:, 1])
from sklearn.metrics import log_loss
log_loss(y_train, grid_search.predict_proba(X_train)[:, 1])
print("best estimator: ", grid_search.best_estimator_)
print("best params: ", grid_search.best_params_)
feat_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train.columns) 
feat_importances.nlargest(20).plot(kind='barh')
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=8,
    min_samples_split=42,
    min_samples_leaf=17
)

model = model.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])
model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']).head())
#complete df_train loss
from sklearn.metrics import log_loss
log_loss(df.loc[df['target'].notna()]['target'], model.predict_proba(df.loc[df['target'].notna()].drop(columns=['target']))[:, 1])
p = model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
sns.distplot(p)
df_submit = pd.DataFrame({
    'uid': df_test['uid'],
    'target': p
})
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
!head /kaggle/working/submit.csv