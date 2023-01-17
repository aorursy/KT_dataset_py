from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression



def variance_inflation_factors(X, clf):

    vifs = []



    for i in range(X.shape[1]):

        sub_X = np.delete(X, i, axis=1)

        sub_y = X[:, i][np.newaxis].T

        sub_clf = clf.fit(sub_X, sub_y)

        sub_y_pred = clf.predict(sub_X)

        

        sub_r2 = r2_score(sub_y, sub_y_pred)

        

        vif = 1 / (1 - sub_r2)

        vifs.append(vif)

        

    return vifs
import numpy as np

from sklearn.linear_model import LinearRegression

clf = LinearRegression()



np.random.seed(42)

X = (np.array(sorted(list(range(5))*20)).reshape(20, 5) +

     np.random.normal(size=100, scale=0.5).reshape(20, 5))

y = (np.array(sorted(list(range(5))*20)).reshape(20, 5) +

     np.random.normal(size=100, scale=0.5).reshape(20, 5))
import seaborn as sns

import pandas as pd

sns.pairplot(pd.DataFrame(X))
variance_inflation_factors(X, clf)
X = (np.array(sorted(list(range(5))*20)).reshape(20, 5) +

     np.random.normal(size=100, scale=1.25).reshape(20, 5))

y = (np.array(sorted(list(range(5))*20)).reshape(20, 5) +

     np.random.normal(size=100, scale=1.25).reshape(20, 5))

sns.pairplot(pd.DataFrame(X))
variance_inflation_factors(X, clf)
import pandas as pd

pd.set_option('max_columns', None)

sales = pd.read_csv("../input/nyc-rolling-sales.csv", index_col=0)

sales.head()
clf = LinearRegression()

X = sales.loc[:, ['RESIDENTIAL UNITS', 'TOTAL UNITS']].dropna().values

variance_inflation_factors(X, clf)
X = (sales.loc[:, ['LAND SQUARE FEET', 'GROSS SQUARE FEET']]

         .replace(' -  ', np.nan)

         .dropna()

         .values

    )

variance_inflation_factors(X, clf)