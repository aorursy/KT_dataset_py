# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import sys,matplotlib,plotly

print("Python version: {}".format(sys.version))
print("NumPy version: {}".format(np.__version__))
print("pandas version: {}".format(pd.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("plotly version:{}".format(plotly.__version__))
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
print(df.info())
df.describe()
def summary(df, pred=None):
    obs=df.shape[0]
    types=df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls=df.apply(lambda x: x.isnull().sum())
    distincts=df.apply(lambda x: x.unique().shape[0])
    missing_ratio=(df.isnull().sum()/ obs)*100
    print('Data Shape: ', df.shape)
    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ratio', 'uniques']
        str = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques], axis = 1)
    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, corr], axis = 1, sort=False)
        corr_col = 'corr '+ pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ratio', 'uniques', corr_col ]
    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n',str.types.value_counts())
    print('___________________________')
    return str
details=summary(df,'target')
details.sort_values(by='missing_ratio', ascending=False)
corr=df.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, cmap='Spectral', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
df1=df.copy()
df1=df1.apply(LabelEncoder().fit_transform)
df1.head()
std_sclr=StandardScaler().fit(df1.drop('target',axis=1))
X=std_sclr.transform(df1.drop('target',axis=1))
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
gbrt=GradientBoostingClassifier(max_depth=1,learning_rate=1,random_state=0)
gbrt.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
param_grid = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
grid_search = GridSearchCV(
    GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
    param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
#grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_
print("Train set score: {:.2f}".format(grid_search.score(X_train, y_train)))
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))
param_grid2 = {'max_depth': [2,3,4,5,6,7]}
grid_search2 = GridSearchCV(
    GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
    param_grid2, scoring='accuracy', cv=5)

grid_search2.fit(X_train, y_train)
#grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_
print("Train set score: {:.2f}".format(grid_search2.score(X_train, y_train)))
print("Test set score: {:.2f}".format(grid_search2.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search2.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search2.best_score_))
print("Best estimator:\n{}".format(grid_search2.best_estimator_))