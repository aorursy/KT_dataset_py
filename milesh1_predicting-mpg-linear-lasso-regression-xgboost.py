# import required libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score
import math
import xgboost
# load and view data
df = pd.read_csv('../input/mpg.csv')
df.head()
# check data types, size of the data, and NA's
print(df.dtypes, end="\n\n")
print('shape of the data is {}. Printing number of NA\'s for each column:'.format(df.shape), end = '\n\n')
print(df.isna().sum())
# features will be every column except the 'MPG Highway' column
features = df.iloc[:,1:].columns.tolist()
# 'MPG Highway' column
target = df.iloc[:,0].name
correlations = {}
for f in features:
    data_temp = df[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
corr = df.corr()
sb.set(style="dark")
sb.set(rc={'figure.figsize':(15,10)})
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap
cmap = sb.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sb.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
sb.countplot(x='MPG Highway', data=df)
plt.show()
print('Standard deviation of {}, Mean of {}'.format(round(np.std(df['MPG Highway']), 2), round(np.mean(df['MPG Highway']), 2)))
regr = linear_model.LinearRegression()
data = df[['Passengers','Length', 'Wheelbase', 'Width','U Turn Space','Rear seat','Luggage','Weight','Horsepower','Fueltank']]
x = data.values
y = df['MPG Highway'].values
for i in range (0,10):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    regr.fit(X_train, y_train)
    print('Score for this round: %0.2f, RMSE: %0.2f' % (regr.score(X_test,y_test), math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2))))
linear_regression_scores = cross_val_score(regr, x, y, cv=10)
print("Accuracy for linear regression: %0.2f (+/- %0.2f)" % (linear_regression_scores.mean(), linear_regression_scores.std() * 2))
print(linear_regression_scores)

clf = linear_model.Lasso()
lasso_regression_scores = cross_val_score(clf, x, y, cv=10)
print("Accuracy for lasso regression: %0.2f (+/- %0.2f)" % (lasso_regression_scores.mean(), lasso_regression_scores.std() * 2))
print(lasso_regression_scores)

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
xgboost_scores = cross_val_score(xgb, x, y, cv=10)
print("Accuracy for XGBoost: %0.2f (+/- %0.2f)" % (xgboost_scores.mean(), xgboost_scores.std() * 2))
print(xgboost_scores)