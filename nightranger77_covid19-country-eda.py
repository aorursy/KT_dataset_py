import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import graphviz

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor, export_graphviz

from sklearn.preprocessing import StandardScaler

from sklearn.externals.six import StringIO  

from IPython.display import Image  
df = pd.read_csv('../input/covid19-demographic-predictors/covid19.csv')
X = df[df.columns[2:]]

y = df['Total Infected']

df[df.columns[1:]].corr()['Total Infected'][:]
plt.scatter(df['GDP 2018'], df['Population 2020'])

plt.xlabel('GDP')

plt.ylabel('Population')

plt.title('GDP vs Population')

plt.show()
df[(df['GDP 2018'] > 1e13) | (df['Population 2020'] > 400000)]
X_s = StandardScaler().fit_transform(X)
est = sm.OLS(y, X_s)

est2 = est.fit()

print(est2.summary(xname=[*X.columns]))

# small pval, reject the null hypothesis that a feature has no effect
rfr = RandomForestRegressor(max_depth=3)

rfr.fit(X_s, y)

print('score', rfr.score(X_s, y), '\n')

[print(c + '\t', f) for c,f in zip(df[df.columns[2:]], rfr.feature_importances_)]
tree = DecisionTreeRegressor(max_depth=3)

tree.fit(X_s, y)

print('score', tree.score(X_s, y), '\n')

print('Feature Importances')

[print(c + '\t', f) for c,f in zip(df[df.columns[2:]], tree.feature_importances_)]
data = export_graphviz(tree, out_file=None,  

                filled=True, rounded=True,

                special_characters=True, feature_names=df.columns[2:])

graph = graphviz.Source(data)

graph
df_small = df[(df['Country'] != 'United States') & (df['Country'] != 'India') & (df['Country'] != 'China')]

X = df_small[df.columns[2:]]

y = df_small['Total Infected']
df_small.sort_values(by='Population 2020', ascending=False).head(5)
X = df_small[df_small.columns[2:]]

y = df_small['Total Infected']

df_small[df_small.columns[1:]].corr()['Total Infected'][:]
plt.scatter(df_small['GDP 2018'], df_small['Population 2020'])

plt.xlabel('GDP')

plt.ylabel('Population')

plt.title('GDP vs Population')

plt.show()
X_s = StandardScaler().fit_transform(X)
est = sm.OLS(y, X_s)

est2 = est.fit()

print(est2.summary(xname=[*X.columns]))

# small pval, reject the null hypothesis that a feature has no effect
rfr = RandomForestRegressor(max_depth=3)

rfr.fit(X_s, y)

print('score', rfr.score(X_s, y), '\n')

[print(c + '\t', f) for c,f in zip(df_small[df_small.columns[2:]], rfr.feature_importances_)]
tree = DecisionTreeRegressor(max_depth=3)

tree.fit(X_s, y)

print('score', tree.score(X_s, y), '\n')

print('Feature Importances')

[print(c + '\t', f) for c,f in zip(df_small[df_small.columns[2:]], tree.feature_importances_)]
data = export_graphviz(tree, out_file=None,  

                filled=True, rounded=True,

                special_characters=True, feature_names=df.columns[2:])

graph = graphviz.Source(data)

graph