# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
df.head()
df.describe()
df.info()
for col in ['city','floor','animal','furniture']:
    print(col,':')
    display(df[col].unique())
df.floor = df.floor.replace('-',0).astype(int)
df['animal'] = df['animal'].map({'acept': 1, 'not acept': 0})
df['furniture'] = df['furniture'].map({'furnished': 1, 'not furnished': 0})
originalNumericColumns = df.select_dtypes(include=np.number).columns.tolist()
df = pd.get_dummies(df)
display(df.head())
plt.figure(figsize=(10,10))
sns.heatmap(df[originalNumericColumns].corr(method='spearman'),annot=True)
df[originalNumericColumns].boxplot(figsize=(10,10), rot=90)
df.boxplot(figsize=(10,10), rot=90, showfliers=False)
from sklearn.ensemble import RandomForestRegressor
# shuffling the data
df = df.sample(frac=1, random_state = 0).reset_index(drop=True)
# removing outliers
q_low = df.quantile(0.005)
q_hi = df.quantile(0.995)
dfsOutliers=[]
for col in df.columns:
    if len(df[col].unique()) < 10: continue # only apply to really numeric attributes        
    print('--------- ',col,' ---------')
    print('Dataframe size before removing outliers rows: ',len(df))
    dfsOutliers.append(df[(df[col] < q_low[col]) | (df[col] > q_hi[col])])
    df = df[(df[col] >= q_low[col]) & (df[col] <= q_hi[col])]
    print('Dataframe size after removing outliers rows: ',len(df))

dfOutliers = pd.concat(dfsOutliers, ignore_index=True)

XCols = list(df.columns)
XCols = [x for x in XCols if x not in ['hoa (R$)','rent amount (R$)','property tax (R$)','fire insurance (R$)','total (R$)']]
yCol = 'total (R$)'
print()
print('XCols: ',XCols)
X = df[XCols].values
y = df[yCol].values
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(XCols, model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: model.criterion+'-importance'})
importances.sort_values(by=model.criterion+'-importance').plot(kind='bar', rot=90)
dfOutliers.head(10)
import collections
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score

models = collections.OrderedDict()
models['KNeighborsRegressor'] = KNeighborsRegressor(n_neighbors=20)
models['LinearRegression'] = LinearRegression()
models['RandomForestRegressor'] = RandomForestRegressor(n_estimators=200, random_state=0)
models['GradientBoostingRegressor'] = GradientBoostingRegressor(n_estimators=200, random_state=0)
models['VotingRegressor'] = VotingRegressor(estimators=[('gb', models['GradientBoostingRegressor']), ('rf', models['RandomForestRegressor']), ('lr', models['LinearRegression'])])
cv = 10
for kModel in models:    
    print('--------- ',kModel,' ---------')
    model = models[kModel]
    scores = cross_val_score(model, X, y, cv=cv, scoring=('r2'))
    display(scores)
    print("scores mean for",kModel,": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
model = models['VotingRegressor']
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
model.fit(X_train, y_train)

y_true = y_test
y_pred = model.predict(X_test)
print('r2_score: ',r2_score(y_true, y_pred))
dfResults = pd.DataFrame()
dfResults['y_true'] = list(y_true)
dfResults['y_pred'] = list(y_pred)
dfResults['%AbsoluteError'] = ((dfResults['y_pred']/dfResults['y_true']-1)*100).abs()
display(dfResults.head(20))
display(dfResults.describe())
Xtest = dfOutliers[XCols].values
ytest = dfOutliers[yCol].values

y_true = ytest
y_pred = model.predict(Xtest)
print('r2_score: ',r2_score(y_true, y_pred))
dfResults = pd.DataFrame()
dfResults['y_true'] = list(y_true)
dfResults['y_pred'] = list(y_pred)
dfResults['%AbsoluteError'] = ((dfResults['y_pred']/dfResults['y_true']-1)*100).abs()
display(dfResults.head(20))
display(dfResults.describe())