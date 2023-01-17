import warnings

warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split,GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/school_fire_cases_1998_2014.csv')

df.columns.values

df['name'] = df['Municipality']
df.head()
plt.hist(df.Population, bins = 20)
plt.hist(df.Population, bins = 20, range=(0,20000))
plt.scatter(x=df.Population, y=df.Cases)
m_simp = pd.read_csv('../input/simplified_municipality_indicators.csv')

m_simp.head()
df = pd.merge(df, m_simp, on="name")
df.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



df = df.select_dtypes(include=numerics)
df.head()
parameters = {

    'n_estimators':[1, 5, 10, 30, 50, 100],

    'min_samples_leaf':[1, 2, 5, 10, 12, 14, 16, 18, 20],

    'max_features':[.2,.5,.8, 1.0]

             }

clf = GridSearchCV(RandomForestRegressor(oob_score=True), parameters)
# train/test split on data

X, test = train_test_split(df, test_size = 0.2)



# pop the classifier off the sets.

y = X.pop('Cases')

test_y = test.pop('Cases')
clf.fit(X,y)
plt.plot(clf.results_['test_mean_score'])
clf.best_score_
sns.barplot(x = clf.best_estimator_.feature_importances_, y = X.columns)

sns.despine(left=True, bottom=True)
clf.best_estimator_
# best_frst = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

#           max_features=0.2, max_leaf_nodes=None, min_impurity_split=1e-07,

#           min_samples_leaf=10, min_samples_split=2,

#           min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,

#           oob_score=True, random_state=None, verbose=0, warm_start=False)
#best_frst.fit(X,y)
#best_frst.oob_score_