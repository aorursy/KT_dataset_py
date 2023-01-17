import statsmodels.api as sm
data = sm.datasets.fair.load_pandas()
data.exog.head()
data.endog.head()
y, X = data.endog, data.exog
X = sm.add_constant(X)
res = sm.OLS(y, X).fit()
res.summary()